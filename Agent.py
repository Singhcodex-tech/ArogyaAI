# ============================================================
# CLINIC AI AGENT — agent.py
# FastAPI server + Groq AI brain + SQLite DB + WhatsApp
# Run: pip install fastapi uvicorn groq httpx python-dotenv
#      uvicorn agent:app --reload
# ============================================================

import os, json, sqlite3, uuid, httpx
from datetime import datetime, date
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Serve index.html at root ────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()

# ── API Keys ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM        = os.getenv("TWILIO_FROM", "whatsapp:+14155238886")  # sandbox number

groq_client  = Groq(api_key=GROQ_API_KEY)

# ── Database ────────────────────────────────────────────────
DB = "clinic.db"

def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    c = db()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT UNIQUE NOT NULL,
            age INTEGER,
            gender TEXT,
            blood_group TEXT,
            address TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS appointments (
            id TEXT PRIMARY KEY,
            patient_id TEXT,
            patient_name TEXT,
            patient_phone TEXT,
            scheduled_at TEXT NOT NULL,
            reason TEXT,
            status TEXT DEFAULT 'scheduled',
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS prescriptions (
            id TEXT PRIMARY KEY,
            patient_id TEXT,
            patient_name TEXT,
            diagnosis TEXT,
            medicines TEXT,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            patient_id TEXT,
            patient_name TEXT,
            patient_phone TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS agent_tasks (
            id TEXT PRIMARY KEY,
            task TEXT,
            status TEXT DEFAULT 'pending',
            result TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    c.commit()
    c.close()

init_db()

# ════════════════════════════════════════════════════════════
# AI AGENT BRAIN
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are ClinicAI, an intelligent medical clinic assistant agent.
You help doctors manage their clinic completely autonomously.

You have access to these tools (respond with JSON when using them):
- get_patients: list all patients
- get_appointments: list appointments for a date
- book_appointment: book an appointment {patient_phone, scheduled_at, reason}
- add_patient: register new patient {name, phone, age, gender}
- write_prescription: generate prescription {patient_id, diagnosis, medicines_list}
- send_whatsapp: send message to patient {phone, message}
- get_stats: get clinic statistics
- answer_patient_query: answer medical/appointment question from patient {query, patient_name}
- cancel_appointment: cancel appointment {appointment_id}
- get_patient_history: get full history of patient {patient_id}

When a doctor gives you an instruction, understand intent and act on it.
When a patient messages, respond with empathy and book/answer as needed.

Always respond in this JSON format:
{
  "thought": "what you understood and plan to do",
  "action": "tool_name or null",
  "action_input": {params} or null,
  "response": "human-friendly response to show",
  "send_whatsapp": true/false,
  "whatsapp_phone": "phone number or null",
  "whatsapp_message": "message to send or null"
}

For prescriptions, medicines should be a list like:
["Paracetamol 500mg - 1 tab twice daily for 5 days", "Cetirizine 10mg - 1 tab at night for 3 days"]

Be proactive. If doctor says 'remind all patients tomorrow', do it.
If patient asks about symptoms, give basic advice and suggest booking.
Always be warm, professional, and efficient."""


PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

def call_groq(messages: list, temperature: float = 0.3) -> str:
    """Call Groq AI with automatic fallback to 8b if 70b fails."""
    last_error = ""
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            resp = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            if model == PRIMARY_MODEL:
                print(f"[ClinicAI] Primary model failed ({last_error}), retrying with fallback {FALLBACK_MODEL}...")
    return json.dumps({"thought": "Error", "action": None, "response": f"AI error: {last_error}", "send_whatsapp": False, "whatsapp_phone": None, "whatsapp_message": None})


def parse_agent_response(raw: str) -> dict:
    """Safely parse agent JSON response."""
    try:
        # Strip markdown fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except:
        return {
            "thought": "Parsed as plain text",
            "action": None,
            "response": raw,
            "send_whatsapp": False,
            "whatsapp_phone": None,
            "whatsapp_message": None
        }


# ════════════════════════════════════════════════════════════
# TOOL EXECUTOR — runs whatever action agent decides
# ════════════════════════════════════════════════════════════

def execute_tool(action: str, params: dict) -> str:
    conn = db()
    try:
        # ── get_stats ──────────────────────────────────────
        if action == "get_stats":
            today = date.today().isoformat()
            total_patients   = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            today_appts      = conn.execute("SELECT COUNT(*) FROM appointments WHERE scheduled_at LIKE ? AND status='scheduled'", (f"{today}%",)).fetchone()[0]
            total_appts      = conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
            total_rx         = conn.execute("SELECT COUNT(*) FROM prescriptions").fetchone()[0]
            return json.dumps({"total_patients": total_patients, "today_appointments": today_appts, "total_appointments": total_appts, "total_prescriptions": total_rx})

        # ── get_patients ───────────────────────────────────
        elif action == "get_patients":
            rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
            return json.dumps([dict(r) for r in rows])

        # ── get_appointments ───────────────────────────────
        elif action == "get_appointments":
            d = params.get("date", date.today().isoformat())
            rows = conn.execute("SELECT * FROM appointments WHERE scheduled_at LIKE ? ORDER BY scheduled_at", (f"{d}%",)).fetchall()
            return json.dumps([dict(r) for r in rows])

        # ── add_patient ────────────────────────────────────
        elif action == "add_patient":
            pid = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO patients (id, name, phone, age, gender) VALUES (?,?,?,?,?)",
                (pid, params.get("name"), params.get("phone"), params.get("age"), params.get("gender"))
            )
            conn.commit()
            return json.dumps({"ok": True, "patient_id": pid, "message": f"Patient {params.get('name')} registered."})

        # ── book_appointment ───────────────────────────────
        elif action == "book_appointment":
            phone = params.get("patient_phone") or params.get("phone")
            patient = conn.execute("SELECT * FROM patients WHERE phone=?", (phone,)).fetchone()
            if not patient:
                return json.dumps({"ok": False, "message": "Patient not found. Register first."})
            aid = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO appointments (id, patient_id, patient_name, patient_phone, scheduled_at, reason) VALUES (?,?,?,?,?,?)",
                (aid, patient["id"], patient["name"], phone, params.get("scheduled_at"), params.get("reason", "General checkup"))
            )
            conn.commit()
            return json.dumps({"ok": True, "appointment_id": aid, "patient": patient["name"], "time": params.get("scheduled_at")})

        # ── cancel_appointment ─────────────────────────────
        elif action == "cancel_appointment":
            conn.execute("UPDATE appointments SET status='cancelled' WHERE id=?", (params.get("appointment_id"),))
            conn.commit()
            return json.dumps({"ok": True, "message": "Appointment cancelled."})

        # ── write_prescription ─────────────────────────────
        elif action == "write_prescription":
            pid  = params.get("patient_id")
            patient = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
            if not patient:
                return json.dumps({"ok": False, "message": "Patient not found."})
            rx_id = str(uuid.uuid4())
            medicines = params.get("medicines_list", params.get("medicines", []))
            if isinstance(medicines, list):
                medicines = "\n".join(medicines)
            conn.execute(
                "INSERT INTO prescriptions (id, patient_id, patient_name, diagnosis, medicines) VALUES (?,?,?,?,?)",
                (rx_id, pid, patient["name"], params.get("diagnosis"), medicines)
            )
            conn.commit()
            return json.dumps({"ok": True, "prescription_id": rx_id, "patient": patient["name"], "diagnosis": params.get("diagnosis"), "medicines": medicines})

        # ── get_patient_history ────────────────────────────
        elif action == "get_patient_history":
            pid = params.get("patient_id")
            patient  = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
            appts    = conn.execute("SELECT * FROM appointments WHERE patient_id=? ORDER BY scheduled_at DESC LIMIT 10", (pid,)).fetchall()
            rx       = conn.execute("SELECT * FROM prescriptions WHERE patient_id=? ORDER BY created_at DESC LIMIT 5", (pid,)).fetchall()
            return json.dumps({"patient": dict(patient) if patient else {}, "appointments": [dict(a) for a in appts], "prescriptions": [dict(r) for r in rx]})

        # ── answer_patient_query ───────────────────────────
        elif action == "answer_patient_query":
            query = params.get("query", "")
            name  = params.get("patient_name", "Patient")
            answer_prompt = [
                {"role": "system", "content": f"You are a helpful clinic assistant. Answer the patient {name}'s query warmly and professionally. Keep it brief. If serious, advise them to visit the doctor."},
                {"role": "user", "content": query}
            ]
            answer = call_groq(answer_prompt, temperature=0.5)
            return json.dumps({"answer": answer})

        # ── send_whatsapp ──────────────────────────────────
        elif action == "send_whatsapp":
            result = send_whatsapp(params.get("phone"), params.get("message", ""))
            return json.dumps(result)

        else:
            return json.dumps({"error": f"Unknown action: {action}"})

    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════
# WHATSAPP VIA TWILIO
# ════════════════════════════════════════════════════════════

async def send_whatsapp_async(phone: str, message: str) -> dict:
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return {"ok": False, "message": "WhatsApp not configured. Add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN to env vars."}
    # Normalize phone → E.164 with country code
    phone = phone.replace("+", "").replace(" ", "").replace("-", "")
    if not phone.startswith("91"):
        phone = "91" + phone
    to_number = f"whatsapp:+{phone}"
    try:
        # Run blocking Twilio client in thread to avoid blocking event loop
        import asyncio
        from twilio.rest import Client
        def _send():
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            msg = client.messages.create(
                from_=TWILIO_FROM,
                body=message,
                to=to_number
            )
            return {"ok": True, "sid": msg.sid, "status": msg.status}
        result = await asyncio.get_event_loop().run_in_executor(None, _send)
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def send_whatsapp(phone: str, message: str) -> dict:
    return await send_whatsapp_async(phone, message)


# ════════════════════════════════════════════════════════════
# MAIN AGENT RUNNER
# ════════════════════════════════════════════════════════════

def run_agent(user_message: str, context: str = "doctor") -> dict:
    """
    Main agent loop:
    1. Send message to Groq AI
    2. Parse action
    3. Execute tool
    4. Feed result back to AI for final response
    5. Send WhatsApp if needed
    """
    # Build context-aware messages
    today_info = f"Today is {datetime.now().strftime('%A, %d %B %Y %I:%M %p')}."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\n{today_info}\nContext: This message is from the {context}."},
        {"role": "user", "content": user_message}
    ]

    # Step 1: Get agent's plan
    raw = call_groq(messages)
    agent_resp = parse_agent_response(raw)

    tool_result = None

    # Step 2: Execute tool if agent decided to use one
    if agent_resp.get("action") and agent_resp["action"] != "null":
        tool_result = execute_tool(agent_resp["action"], agent_resp.get("action_input") or {})

        # Step 3: Feed tool result back for final natural language response
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Tool result: {tool_result}\n\nNow give me the final response to the user based on this result."})
        raw2 = call_groq(messages)
        agent_resp2 = parse_agent_response(raw2)
        # Keep original action info but use new response
        agent_resp["response"] = agent_resp2.get("response", agent_resp.get("response"))
        if agent_resp2.get("send_whatsapp"):
            agent_resp["send_whatsapp"]     = True
            agent_resp["whatsapp_phone"]    = agent_resp2.get("whatsapp_phone")
            agent_resp["whatsapp_message"]  = agent_resp2.get("whatsapp_message")

    # Step 4: Send WhatsApp if agent decided to
    whatsapp_sent = False
    if agent_resp.get("send_whatsapp") and agent_resp.get("whatsapp_phone") and agent_resp.get("whatsapp_message"):
        wa_result = send_whatsapp(agent_resp["whatsapp_phone"], agent_resp["whatsapp_message"])
        whatsapp_sent = wa_result.get("ok", False)

    # Step 5: Log task
    conn = db()
    conn.execute(
        "INSERT INTO agent_tasks (id, task, status, result) VALUES (?,?,?,?)",
        (str(uuid.uuid4()), user_message, "completed", agent_resp.get("response", ""))
    )
    conn.commit()
    conn.close()

    return {
        "thought":        agent_resp.get("thought", ""),
        "response":       agent_resp.get("response", "Done."),
        "action":         agent_resp.get("action"),
        "tool_result":    json.loads(tool_result) if tool_result else None,
        "whatsapp_sent":  whatsapp_sent,
    }


# ════════════════════════════════════════════════════════════
# API ROUTES — called by index.html
# ════════════════════════════════════════════════════════════

@app.post("/agent")
async def agent_endpoint(req: Request):
    """Main AI agent endpoint — doctor talks to agent here."""
    body    = await req.json()
    message = body.get("message", "")
    context = body.get("context", "doctor")
    if not message:
        return JSONResponse({"error": "No message"}, status_code=400)
    result = run_agent(message, context)
    return JSONResponse(result)


@app.post("/patient-message")
async def patient_message(req: Request):
    """Patient sends a message — agent handles it and replies via WhatsApp."""
    body    = await req.json()
    phone   = body.get("phone", "")
    message = body.get("message", "")
    name    = body.get("name", "Patient")

    # Save incoming message
    conn = db()
    patient = conn.execute("SELECT * FROM patients WHERE phone=?", (phone,)).fetchone()
    pid = patient["id"] if patient else None
    conn.execute(
        "INSERT INTO messages (id, patient_id, patient_name, patient_phone, role, content) VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()), pid, name, phone, "patient", message)
    )
    conn.commit()
    conn.close()

    # Let agent handle it
    full_message = f"Patient {name} (phone: {phone}) says: {message}"
    result = run_agent(full_message, context="patient")

    # Auto-reply via WhatsApp
    if result.get("response") and phone:
        send_whatsapp(phone, result["response"])

    return JSONResponse({"reply": result["response"], "whatsapp_sent": True})


@app.get("/api/stats")
async def get_stats():
    conn = db()
    today = date.today().isoformat()
    stats = {
        "total_patients":      conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0],
        "today_appointments":  conn.execute("SELECT COUNT(*) FROM appointments WHERE scheduled_at LIKE ? AND status='scheduled'", (f"{today}%",)).fetchone()[0],
        "total_appointments":  conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0],
        "total_prescriptions": conn.execute("SELECT COUNT(*) FROM prescriptions").fetchone()[0],
    }
    conn.close()
    return JSONResponse(stats)


@app.get("/api/patients")
async def get_patients():
    conn = db()
    rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/appointments")
async def get_appointments(date: str = None):
    conn = db()
    d = date or datetime.now().date().isoformat()
    rows = conn.execute("SELECT * FROM appointments WHERE scheduled_at LIKE ? ORDER BY scheduled_at", (f"{d}%",)).fetchall()
    conn.close()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/prescriptions")
async def get_prescriptions():
    conn = db()
    rows = conn.execute("SELECT * FROM prescriptions ORDER BY created_at DESC LIMIT 20").fetchall()
    conn.close()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/messages")
async def get_messages():
    conn = db()
    rows = conn.execute("SELECT * FROM messages ORDER BY created_at DESC LIMIT 50").fetchall()
    conn.close()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/tasks")
async def get_tasks():
    conn = db()
    rows = conn.execute("SELECT * FROM agent_tasks ORDER BY created_at DESC LIMIT 20").fetchall()
    conn.close()
    return JSONResponse([dict(r) for r in rows])


@app.post("/api/patients")
async def add_patient(req: Request):
    body = await req.json()
    pid  = str(uuid.uuid4())
    conn = db()
    try:
        conn.execute(
            "INSERT INTO patients (id, name, phone, age, gender, blood_group, address) VALUES (?,?,?,?,?,?,?)",
            (pid, body["name"], body["phone"], body.get("age"), body.get("gender"), body.get("blood_group"), body.get("address"))
        )
        conn.commit()
        return JSONResponse({"ok": True, "id": pid})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    finally:
        conn.close()


@app.post("/api/whatsapp/send")
async def manual_whatsapp(req: Request):
    body   = await req.json()
    result = await send_whatsapp_async(body.get("phone"), body.get("message"))
    return JSONResponse(result)