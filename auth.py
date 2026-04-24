import hashlib
import secrets
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

DB_URL = "postgresql://postgres:016758.DAfe*@db.isgumzwugaibqnqeevwd.supabase.co:5432/postgres"

GMAIL_USER = "adminfs01@gmail.com"
GMAIL_PASS = "izdq scag mtgy oulz"

ADMIN_EMAIL = "adminfs01@gmail.com"
ADMIN_USER  = "admin"
ADMIN_PASS  = "FruitScanFR1728"

def get_db():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = get_db()
    c = conn.cursor()

    # Tabla usuarios
    c.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            verificado BOOLEAN DEFAULT FALSE,
            token TEXT,
            token_expiry TIMESTAMP,
            creado_en TIMESTAMP DEFAULT NOW()
        )
    """)

    # Tabla predicciones
    c.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id SERIAL PRIMARY KEY,
            usuario_id INTEGER REFERENCES usuarios(id) ON DELETE CASCADE,
            modelo TEXT NOT NULL,
            fruta TEXT NOT NULL,
            estado TEXT NOT NULL,
            confianza_fruta FLOAT NOT NULL,
            confianza_estado FLOAT NOT NULL,
            fecha TIMESTAMP DEFAULT NOW()
        )
    """)

    # Tabla estadisticas por modelo
    c.execute("""
        CREATE TABLE IF NOT EXISTS estadisticas_modelo (
            id SERIAL PRIMARY KEY,
            modelo TEXT NOT NULL,
            total_predicciones INTEGER DEFAULT 0,
            frutas_correctas INTEGER DEFAULT 0,
            ultima_vez TIMESTAMP DEFAULT NOW(),
            UNIQUE(modelo)
        )
    """)

    # Insertar modelos base en estadisticas
    for modelo in ["InceptionV3", "MobileNetV2", "VGG16"]:
        c.execute("""
            INSERT INTO estadisticas_modelo (modelo)
            VALUES (%s)
            ON CONFLICT (modelo) DO NOTHING
        """, (modelo,))

    # Crear admin si no existe
    c.execute("SELECT id FROM usuarios WHERE email = %s", (ADMIN_EMAIL,))
    if not c.fetchone():
        c.execute("""
            INSERT INTO usuarios (username, email, password_hash, verificado)
            VALUES (%s, %s, %s, TRUE)
        """, (ADMIN_USER, ADMIN_EMAIL, hash_password(ADMIN_PASS)))

    conn.commit()
    conn.close()

def enviar_token(email: str, token: str) -> bool:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "FruitScan — Verifica tu cuenta"
        msg["From"]    = GMAIL_USER
        msg["To"]      = email

        html = f"""
        <div style="font-family:sans-serif;max-width:480px;margin:0 auto;background:#111826;
                    border-radius:16px;overflow:hidden;border:1px solid #1e2d42">
            <div style="background:linear-gradient(90deg,#38bdf8,#818cf8);height:4px"></div>
            <div style="padding:36px 32px">
                <div style="font-size:1.6rem;font-weight:900;color:#38bdf8;margin-bottom:4px">
                    🍊 FruitScan
                </div>
                <div style="font-size:.75rem;color:#4a6070;letter-spacing:.15em;
                            text-transform:uppercase;margin-bottom:28px">
                    Verificación de cuenta
                </div>
                <p style="color:#94a8bc;font-size:.95rem;line-height:1.7;margin-bottom:24px">
                    Hola, usa el siguiente token para verificar tu cuenta.
                    Este código expira en <strong style="color:#e8f0f8">15 minutos</strong>.
                </p>
                <div style="background:#080c12;border:1px solid #1e2d42;border-radius:12px;
                            padding:20px;text-align:center;margin-bottom:24px">
                    <div style="font-family:monospace;font-size:2rem;font-weight:700;
                                letter-spacing:.3em;color:#38bdf8">{token}</div>
                </div>
                <p style="color:#4a6070;font-size:.8rem;line-height:1.6">
                    Si no creaste esta cuenta, ignora este correo.
                </p>
            </div>
        </div>
        """
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.sendmail(GMAIL_USER, email, msg.as_string())
        return True
    except Exception as e:
        print(f"Error enviando correo: {e}")
        return False

def registrar_usuario(username: str, email: str, password: str) -> dict:
    conn = get_db()
    c    = conn.cursor()
    c.execute("SELECT id FROM usuarios WHERE email = %s", (email,))
    if c.fetchone():
        conn.close()
        return {"ok": False, "msg": "Este correo ya está registrado."}
    c.execute("SELECT id FROM usuarios WHERE username = %s", (username,))
    if c.fetchone():
        conn.close()
        return {"ok": False, "msg": "Este nombre de usuario ya está en uso."}

    token  = str(secrets.randbelow(900000) + 100000)
    expiry = datetime.now() + timedelta(minutes=15)

    c.execute("""
        INSERT INTO usuarios (username, email, password_hash, token, token_expiry)
        VALUES (%s, %s, %s, %s, %s)
    """, (username, email, hash_password(password), token, expiry))
    conn.commit()
    conn.close()

    enviado = enviar_token(email, token)
    if enviado:
        return {"ok": True,  "msg": f"Token enviado a {email}. Revisa tu bandeja."}
    else:
        return {"ok": False, "msg": "Error enviando el correo. Verifica el email ingresado."}

def verificar_token(email: str, token: str) -> dict:
    conn = get_db()
    c    = conn.cursor()
    c.execute("SELECT * FROM usuarios WHERE email = %s", (email,))
    user = c.fetchone()
    if not user:
        conn.close()
        return {"ok": False, "msg": "Usuario no encontrado."}
    if user["verificado"]:
        conn.close()
        return {"ok": True, "msg": "Cuenta ya verificada."}
    if user["token"] != token:
        conn.close()
        return {"ok": False, "msg": "Token incorrecto."}
    if datetime.now() > user["token_expiry"]:
        conn.close()
        return {"ok": False, "msg": "Token expirado. Regístrate de nuevo."}
    c.execute("UPDATE usuarios SET verificado=TRUE, token=NULL, token_expiry=NULL WHERE email=%s", (email,))
    conn.commit()
    conn.close()
    return {"ok": True, "msg": "Cuenta verificada exitosamente."}

def login_usuario(email: str, password: str) -> dict:
    conn = get_db()
    c    = conn.cursor()
    c.execute("SELECT * FROM usuarios WHERE email = %s", (email,))
    user = c.fetchone()
    conn.close()
    if not user:
        return {"ok": False, "msg": "Correo no registrado."}
    if not user["verificado"]:
        return {"ok": False, "msg": "Cuenta no verificada. Revisa tu correo."}
    if user["password_hash"] != hash_password(password):
        return {"ok": False, "msg": "Contraseña incorrecta."}
    return {"ok": True, "username": user["username"], "email": user["email"], "id": user["id"]}

def guardar_prediccion(usuario_id: int, modelo: str, fruta: str, estado: str,
                        confianza_fruta: float, confianza_estado: float):
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        INSERT INTO predicciones (usuario_id, modelo, fruta, estado, confianza_fruta, confianza_estado)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (usuario_id, modelo, fruta, estado, confianza_fruta, confianza_estado))
    c.execute("""
        UPDATE estadisticas_modelo
        SET total_predicciones = total_predicciones + 1,
            ultima_vez = NOW()
        WHERE modelo = %s
    """, (modelo,))
    conn.commit()
    conn.close()

def obtener_historial(usuario_id: int, limite: int = 20) -> list:
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        SELECT modelo, fruta, estado, confianza_fruta, confianza_estado, fecha
        FROM predicciones
        WHERE usuario_id = %s
        ORDER BY fecha DESC
        LIMIT %s
    """, (usuario_id, limite))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def obtener_estadisticas() -> list:
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        SELECT
            e.modelo,
            e.total_predicciones,
            e.ultima_vez,
            COUNT(p.id) FILTER (WHERE p.estado = 'Fresca') AS total_frescas,
            COUNT(p.id) FILTER (WHERE p.estado = 'Podrida') AS total_podridas,
            ROUND(AVG(p.confianza_fruta)::numeric * 100, 1) AS avg_confianza_fruta,
            ROUND(AVG(p.confianza_estado)::numeric * 100, 1) AS avg_confianza_estado
        FROM estadisticas_modelo e
        LEFT JOIN predicciones p ON p.modelo = e.modelo
        GROUP BY e.modelo, e.total_predicciones, e.ultima_vez
        ORDER BY e.total_predicciones DESC
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def obtener_resumen_frutas() -> list:
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        SELECT fruta, COUNT(*) as total,
               ROUND(AVG(confianza_fruta)::numeric * 100, 1) as avg_confianza
        FROM predicciones
        GROUP BY fruta
        ORDER BY total DESC
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]
