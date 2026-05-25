import os
import io
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import pandas as pd
from auth import (
    init_db, registrar_usuario, verificar_token, login_usuario,
    guardar_prediccion, obtener_historial, obtener_estadisticas,
    obtener_resumen_frutas, obtener_usuarios, cambiar_rol,
    toggle_usuario_activo, obtener_logs, registrar_log,
    tiene_permiso, ROLES,
)

st.set_page_config(
    page_title="FruitScan",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("tema",      "dark"),
    ("usuario",   None),
    ("pantalla",  "login"),
    ("reg_email", ""),
    ("tab",       "clasificar"),
]:
    if k not in st.session_state:
        st.session_state[k] = v

tema = st.session_state.tema

if tema == "dark":
    BG="#080c12"; SURFACE="#111826"; BORDER="#1e2d42"
    TEXT="#e8f0f8"; TEXT2="#94a8bc"; MUTED="#4a6070"
    ACCENT="#38bdf8"; ACCENT2="#818cf8"; FRESH="#34d399"; ROTTEN="#fb7185"
    GRAD1="rgba(56,189,248,.06)"; GRAD2="rgba(129,140,248,.06)"
    WARN="#fbbf24"; DANGER="#f87171"
else:
    BG="#f0f5fb"; SURFACE="#ffffff"; BORDER="#d0dce8"
    TEXT="#0d1926"; TEXT2="#3d5470"; MUTED="#8099b0"
    ACCENT="#0284c7"; ACCENT2="#6366f1"; FRESH="#059669"; ROTTEN="#e11d48"
    GRAD1="rgba(2,132,199,.05)"; GRAD2="rgba(99,102,241,.05)"
    WARN="#d97706"; DANGER="#dc2626"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{{font-family:'Outfit',sans-serif!important}}
[data-testid="stAppViewContainer"]{{background:{BG};background-image:radial-gradient(ellipse 60% 40% at 10% 10%,{GRAD1} 0%,transparent 60%),radial-gradient(ellipse 40% 30% at 90% 80%,{GRAD2} 0%,transparent 60%)}}
[data-testid="stSidebar"]{{background:{SURFACE}!important;border-right:1px solid {BORDER}}}
[data-testid="stHeader"]{{background:transparent}}
.main-title{{font-family:'Outfit',sans-serif;font-weight:900;font-size:3.2rem;letter-spacing:-.03em;color:{ACCENT};line-height:1}}
.main-title span{{color:{TEXT}}}
.main-subtitle{{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:{MUTED};margin-bottom:2rem}}
.auth-card{{background:{SURFACE};border:1px solid {BORDER};border-radius:20px;padding:40px;max-width:460px;margin:0 auto;box-shadow:0 8px 40px rgba(0,0,0,.2);position:relative;overflow:hidden}}
.auth-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{ACCENT},{ACCENT2})}}
.auth-title{{font-family:'Outfit',sans-serif;font-weight:900;font-size:1.6rem;color:{TEXT};margin-bottom:4px}}
.auth-sub{{font-size:.8rem;color:{MUTED};margin-bottom:28px}}
.divider{{text-align:center;color:{MUTED};font-size:.8rem;margin:16px 0}}
.result-card{{background:{SURFACE};border:1px solid {BORDER};border-radius:16px;padding:28px 32px;margin-top:8px;position:relative;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.15)}}
.result-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{ACCENT},{ACCENT2})}}
.result-fruta{{font-family:'Outfit',sans-serif;font-size:2.6rem;font-weight:900;letter-spacing:-.02em;color:{TEXT};line-height:1;margin-bottom:10px}}
.result-estado-fresh{{font-size:1rem;font-weight:600;color:{FRESH};background:rgba(52,211,153,.1);border:1px solid {FRESH};display:inline-block;padding:5px 18px;border-radius:100px;margin-bottom:20px}}
.result-estado-rotten{{font-size:1rem;font-weight:600;color:{ROTTEN};background:rgba(251,113,133,.1);border:1px solid {ROTTEN};display:inline-block;padding:5px 18px;border-radius:100px;margin-bottom:20px}}
.conf-row{{display:flex;gap:16px;margin-top:10px}}
.conf-box{{flex:1;background:{BG};border:1px solid {BORDER};border-radius:12px;padding:14px 18px}}
.conf-label{{font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:{MUTED};margin-bottom:4px;font-weight:600}}
.conf-value{{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:500;color:{ACCENT}}}
.bar-section-title{{font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:{MUTED};margin-bottom:12px;margin-top:20px;font-weight:600}}
.bar-row-c{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.bar-name{{width:120px;font-size:.76rem;color:{TEXT2};flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.bar-track{{flex:1;height:6px;background:{BORDER};border-radius:4px;overflow:hidden}}
.bar-fill-a{{height:100%;border-radius:4px;background:linear-gradient(90deg,{ACCENT},{ACCENT2})}}
.bar-fill-b{{height:100%;border-radius:4px;background:{BORDER}}}
.bar-fill-f{{height:100%;border-radius:4px;background:{FRESH}}}
.bar-fill-r{{height:100%;border-radius:4px;background:{ROTTEN}}}
.bar-pct{{width:48px;font-size:.7rem;color:{MUTED};text-align:right;flex-shrink:0;font-family:'JetBrains Mono',monospace}}
.model-tag{{font-size:.65rem;color:{MUTED};letter-spacing:.12em;text-transform:uppercase;margin-top:14px;font-family:'JetBrains Mono',monospace}}
.stat-card{{background:{SURFACE};border:1px solid {BORDER};border-radius:14px;padding:20px 24px;text-align:center}}
.stat-value{{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;color:{ACCENT}}}
.stat-label{{font-size:.65rem;letter-spacing:.15em;text-transform:uppercase;color:{MUTED};margin-top:4px;font-weight:600}}
.hist-row{{background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:14px 18px;margin-bottom:8px;display:flex;align-items:center;gap:16px}}
.user-badge{{display:inline-flex;align-items:center;gap:8px;background:{BG};border:1px solid {BORDER};border-radius:100px;padding:6px 14px;font-size:.8rem;color:{TEXT2}}}
.rol-badge{{font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;padding:3px 10px;border-radius:100px;font-weight:700;display:inline-block}}
.rol-admin{{background:rgba(251,191,36,.12);color:{WARN};border:1px solid {WARN}}}
.rol-investigador{{background:rgba(129,140,248,.12);color:{ACCENT2};border:1px solid {ACCENT2}}}
.rol-usuario{{background:rgba(56,189,248,.10);color:{ACCENT};border:1px solid {ACCENT}}}
.log-row{{background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:10px 16px;margin-bottom:6px;font-size:.75rem;font-family:'JetBrains Mono',monospace;color:{TEXT2}}}
.log-warn{{border-left:3px solid {WARN}}}
.log-error{{border-left:3px solid {DANGER}}}
.log-info{{border-left:3px solid {ACCENT}}}
.acceso-denegado{{background:{SURFACE};border:2px solid {DANGER};border-radius:16px;padding:40px;text-align:center;color:{DANGER};font-size:1.1rem;font-weight:700}}
#MainMenu,footer{{visibility:hidden}}
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
MODELOS_DRIVE = {
    "InceptionV3": "1BYrUWEhK_4NtXY_SKH2mpKTJ_uusHYOT",
    "MobileNetV2": "1MGM1f8c46f07j1K_UUllfoplhjCBhQCp",
    "VGG16":       "1g7EhijZb7zL_VxwEPYeJ54Sny-bhII1m",
}
CLASES_FRUTA  = ["Banano","Fresa","Limón","Lulo","Mango","Naranja","Tomate","Tomate de Árbol"]
CLASES_ESTADO = ["Fresca","Podrida"]
EMOJIS_FRUTA  = {"Banano":"🍌","Fresa":"🍓","Limón":"🍋","Lulo":"🟠","Mango":"🥭","Naranja":"🍊","Tomate":"🍅","Tomate de Árbol":"🍅"}
MODELOS_DIR   = "/tmp/modelos"
os.makedirs(MODELOS_DIR, exist_ok=True)

# Validación de imágenes (8.4)
MAX_FILE_SIZE_MB  = 5
TIPOS_PERMITIDOS  = {"jpeg", "png", "webp"}

def _detectar_tipo(datos: bytes) -> str | None:
    """Detecta el tipo de imagen por magic bytes, sin dependencias externas."""
    if datos[:3] == b'\xff\xd8\xff':
        return "jpeg"
    if datos[:8] == b'\x89PNG\r\n\x1a\n':
        return "png"
    if datos[:4] in (b'RIFF',) and datos[8:12] == b'WEBP':
        return "webp"
    return None

def validar_imagen(archivo) -> tuple[bool, str]:
    """Valida tipo real por magic bytes, tamaño e integridad con PIL."""
    datos = archivo.read()
    archivo.seek(0)

    # Tamaño
    size_mb = len(datos) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"La imagen supera el tamaño máximo de {MAX_FILE_SIZE_MB} MB ({size_mb:.1f} MB)."

    # Tipo real por magic bytes
    tipo_real = _detectar_tipo(datos)
    if tipo_real not in TIPOS_PERMITIDOS:
        return False, "Tipo de archivo no permitido. Sube una imagen JPG, PNG o WebP."

    # Integridad con PIL
    try:
        img = Image.open(io.BytesIO(datos))
        img.verify()
    except Exception:
        return False, "El archivo está dañado o no es una imagen válida."

    return True, ""

@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo(nombre):
    ruta = os.path.join(MODELOS_DIR, f"{nombre}.tflite")
    if not os.path.exists(ruta):
        gdown.download(f"https://drive.google.com/uc?id={MODELOS_DRIVE[nombre]}", ruta, quiet=False)
    interpreter = tf.lite.Interpreter(model_path=ruta)
    interpreter.allocate_tensors()
    return interpreter

def predecir(interpreter, imagen):
    img = imagen.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    interpreter.set_tensor(inp[0]["index"], arr)
    interpreter.invoke()
    salidas = [interpreter.get_tensor(o["index"])[0] for o in out]
    return salidas[0], salidas[1]

def badge_rol(rol: str) -> str:
    cls = {"admin": "rol-admin", "investigador": "rol-investigador"}.get(rol, "rol-usuario")
    return f'<span class="rol-badge {cls}">{ROLES.get(rol, rol)}</span>'

# ══════════════════════════════════════════════════════════════════════════════
# PANTALLAS AUTH
# ══════════════════════════════════════════════════════════════════════════════
def pantalla_login():
    st.markdown('<div class="main-title">FRUIT<span>SCAN</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Clasificador multitarea · Fruta & Estado</div>', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Iniciar sesión</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Ingresa con tu cuenta de FruitScan</div>', unsafe_allow_html=True)
        email    = st.text_input("Correo electrónico", placeholder="usuario@email.com", key="login_email")
        password = st.text_input("Contraseña", type="password", placeholder="••••••••", key="login_pass")
        if st.button("Ingresar", use_container_width=True, type="primary"):
            if not email or not password:
                st.error("Completa todos los campos.")
            else:
                res = login_usuario(email, password)
                if res["ok"]:
                    st.session_state.usuario  = {
                        "username": res["username"],
                        "email":    res["email"],
                        "id":       res["id"],
                        "rol":      res["rol"],
                    }
                    st.session_state.pantalla = "app"
                    st.session_state.tab      = "clasificar"
                    st.rerun()
                else:
                    st.error(res["msg"])
        st.markdown('<div class="divider">¿No tienes cuenta?</div>', unsafe_allow_html=True)
        if st.button("Crear cuenta nueva", use_container_width=True):
            st.session_state.pantalla = "registro"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def pantalla_registro():
    st.markdown('<div class="main-title">FRUIT<span>SCAN</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Crear cuenta nueva</div>', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Registro</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Te enviaremos un token de verificación a tu correo</div>', unsafe_allow_html=True)
        username  = st.text_input("Nombre de usuario", placeholder="ej: juan123", key="reg_user")
        email     = st.text_input("Correo electrónico", placeholder="usuario@email.com", key="reg_mail")
        password  = st.text_input("Contraseña", type="password", placeholder="Mínimo 8 caracteres", key="reg_pass")
        password2 = st.text_input("Confirmar contraseña", type="password", placeholder="Repite la contraseña", key="reg_pass2")
        rol       = st.selectbox("Rol", options=["usuario", "investigador"],
                                 format_func=lambda r: ROLES[r], key="reg_rol")
        if st.button("Registrarme", use_container_width=True, type="primary"):
            if not all([username, email, password, password2]):
                st.error("Completa todos los campos.")
            elif len(password) < 8:
                st.error("La contraseña debe tener al menos 8 caracteres.")
            elif password != password2:
                st.error("Las contraseñas no coinciden.")
            else:
                with st.spinner("Enviando token..."):
                    res = registrar_usuario(username, email, password, rol)
                if res["ok"]:
                    st.session_state.reg_email = email
                    st.session_state.pantalla  = "verificar"
                    st.rerun()
                else:
                    st.error(res["msg"])
        st.markdown('<div class="divider">¿Ya tienes cuenta?</div>', unsafe_allow_html=True)
        if st.button("Iniciar sesión", use_container_width=True):
            st.session_state.pantalla = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def pantalla_verificar():
    st.markdown('<div class="main-title">FRUIT<span>SCAN</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Verifica tu cuenta</div>', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Código de verificación</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="auth-sub">Token enviado a <strong>{st.session_state.reg_email}</strong></div>', unsafe_allow_html=True)
        token = st.text_input("Token de 6 dígitos", placeholder="123456", max_chars=6, key="token_input")
        if st.button("Verificar cuenta", use_container_width=True, type="primary"):
            if not token:
                st.error("Ingresa el token.")
            else:
                res = verificar_token(st.session_state.reg_email, token)
                if res["ok"]:
                    st.success("✅ Cuenta verificada. Ya puedes iniciar sesión.")
                    st.session_state.pantalla = "login"
                    st.rerun()
                else:
                    st.error(res["msg"])
        st.markdown('<div class="divider">¿Token vencido?</div>', unsafe_allow_html=True)
        if st.button("Volver al registro", use_container_width=True):
            st.session_state.pantalla = "registro"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# APP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def pantalla_app():
    u   = st.session_state.usuario
    rol = u["rol"]

    with st.sidebar:
        st.markdown('<div class="main-title" style="font-size:2rem">FRUIT<span>SCAN</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="main-subtitle">Clasificador de frutas</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="user-badge">👤 {u["username"]} &nbsp;{badge_rol(rol)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        icono = "☀️" if tema == "dark" else "🌙"
        label = "Modo claro" if tema == "dark" else "Modo oscuro"
        if st.button(f"{icono} {label}", use_container_width=True):
            st.session_state.tema = "light" if tema == "dark" else "dark"
            st.rerun()
        if st.button("🚪 Cerrar sesión", use_container_width=True):
            registrar_log("LOGOUT", f"Sesión cerrada: {u['username']}", usuario_id=u["id"])
            st.session_state.usuario  = None
            st.session_state.pantalla = "login"
            st.rerun()

        st.markdown("---")
        st.markdown(f'<div style="font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:{MUTED};margin-bottom:10px;font-weight:600">Modelo</div>', unsafe_allow_html=True)
        nombre_modelo = st.selectbox("", list(MODELOS_DRIVE.keys()), index=0, label_visibility="collapsed")
        st.markdown("---")
        st.markdown(f'<div style="font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:{MUTED};margin-bottom:10px;font-weight:600">Navegación</div>', unsafe_allow_html=True)

        # Botones según permisos del rol
        if tiene_permiso(rol, "clasificar"):
            if st.button("🔍 Clasificar", use_container_width=True):
                st.session_state.tab = "clasificar"; st.rerun()
        if tiene_permiso(rol, "historial"):
            if st.button("📋 Mi historial", use_container_width=True):
                st.session_state.tab = "historial"; st.rerun()
        if tiene_permiso(rol, "estadisticas"):
            if st.button("📊 Estadísticas", use_container_width=True):
                st.session_state.tab = "estadisticas"; st.rerun()
        if tiene_permiso(rol, "panel_admin"):
            st.markdown("---")
            st.markdown(f'<div style="font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:{WARN};margin-bottom:10px;font-weight:600">Admin</div>', unsafe_allow_html=True)
            if st.button("👥 Usuarios", use_container_width=True):
                st.session_state.tab = "panel_admin"; st.rerun()
            if st.button("📝 Logs", use_container_width=True):
                st.session_state.tab = "ver_logs"; st.rerun()

    tab = st.session_state.tab

    # ── Verificación de acceso por rol ────────────────────────────────────────
    if not tiene_permiso(rol, tab):
        st.markdown(
            f'<div class="acceso-denegado">🔒 Acceso denegado<br>'
            f'<span style="font-size:.8rem;font-weight:400;color:{MUTED}">Tu rol ({ROLES[rol]}) no tiene permiso para esta sección.</span></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Tab Clasificar ────────────────────────────────────────────────────────
    if tab == "clasificar":
        st.markdown('<div class="main-title">FRUIT<span>SCAN</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="main-subtitle">Clasificador multitarea · Fruta & Estado</div>', unsafe_allow_html=True)

        col_img, col_res = st.columns([1.1, 1], gap="large")

        with col_img:
            if st.session_state.get("limpiar"):
                st.session_state["limpiar"] = False
                st.rerun()
            archivo = st.file_uploader(
                f"Sube una imagen (JPG/PNG/WebP, máx {MAX_FILE_SIZE_MB} MB)",
                type=["jpg", "jpeg", "png", "webp"],
                key="uploader",
            )
            if archivo:
                # ── Validación 8.4 ────────────────────────────────────────
                valido, msg_error = validar_imagen(archivo)
                if not valido:
                    st.error(f"⚠️ {msg_error}")
                    registrar_log("IMAGEN_INVALIDA", msg_error, usuario_id=u["id"], nivel="WARNING")
                else:
                    imagen = Image.open(archivo)
                    st.image(imagen, use_container_width=True)
                    if st.button("🗑️ Quitar imagen", use_container_width=True):
                        st.session_state["limpiar"] = True
                        st.rerun()

                    with col_res:
                        with st.spinner("Analizando..."):
                            try:
                                interpreter = cargar_modelo(nombre_modelo)
                                pred_fruta, pred_estado = predecir(interpreter, imagen)
                                fruta   = CLASES_FRUTA[int(np.argmax(pred_fruta))]
                                estado  = CLASES_ESTADO[int(np.argmax(pred_estado))]
                                emoji   = EMOJIS_FRUTA.get(fruta, "🍑")
                                conf_f  = float(np.max(pred_fruta)) * 100
                                conf_e  = float(np.max(pred_estado)) * 100
                                fresca  = estado == "Fresca"
                                icono_e = "✅" if fresca else "🔴"
                                ecls    = "result-estado-fresh" if fresca else "result-estado-rotten"

                                guardar_prediccion(u["id"], nombre_modelo, fruta, estado,
                                                   conf_f / 100, conf_e / 100)

                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="result-fruta">{emoji} {fruta}</div>
                                    <div class="{ecls}">{icono_e} {estado}</div>
                                    <div class="conf-row">
                                        <div class="conf-box"><div class="conf-label">Confianza fruta</div><div class="conf-value">{conf_f:.1f}%</div></div>
                                        <div class="conf-box"><div class="conf-label">Confianza estado</div><div class="conf-value">{conf_e:.1f}%</div></div>
                                    </div>
                                    <div class="model-tag">Modelo: {nombre_modelo} finetuned</div>
                                </div>""", unsafe_allow_html=True)

                                st.markdown('<div class="bar-section-title">Probabilidad por fruta</div>', unsafe_allow_html=True)
                                for n, p in sorted(zip(CLASES_FRUTA, pred_fruta), key=lambda x: x[1], reverse=True):
                                    pct  = float(p) * 100
                                    fill = "bar-fill-a" if n == fruta else "bar-fill-b"
                                    st.markdown(f'<div class="bar-row-c"><span class="bar-name">{n}</span><div class="bar-track"><div class="{fill}" style="width:{pct:.1f}%"></div></div><span class="bar-pct">{pct:.1f}%</span></div>', unsafe_allow_html=True)

                                st.markdown('<div class="bar-section-title">Estado</div>', unsafe_allow_html=True)
                                for n, p in zip(CLASES_ESTADO, pred_estado):
                                    pct  = float(p) * 100
                                    fill = "bar-fill-f" if n == "Fresca" else "bar-fill-r"
                                    st.markdown(f'<div class="bar-row-c"><span class="bar-name">{n}</span><div class="bar-track"><div class="{fill}" style="width:{pct:.1f}%"></div></div><span class="bar-pct">{pct:.1f}%</span></div>', unsafe_allow_html=True)

                            except Exception as e:
                                registrar_log("ERROR_PREDICCION", str(e), usuario_id=u["id"], nivel="ERROR")
                                st.error(f"Error al procesar: {e}")
            else:
                st.markdown(f"""
                <div style="aspect-ratio:1;background:{SURFACE};border:2px dashed {BORDER};
                     border-radius:16px;display:flex;align-items:center;justify-content:center;
                     flex-direction:column;gap:12px;color:{MUTED};text-align:center;padding:40px">
                    <div style="font-size:3rem;filter:grayscale(.5)">🍊</div>
                    <div style="font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;line-height:1.8">
                        Arrastra o selecciona<br>una imagen<br>
                        <span style="font-size:.65rem;opacity:.6">JPG · PNG · WebP · máx {MAX_FILE_SIZE_MB} MB</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Tab Historial ─────────────────────────────────────────────────────────
    elif tab == "historial":
        st.markdown('<div class="main-title">Mi <span>Historial</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="main-subtitle">Últimas 20 predicciones</div>', unsafe_allow_html=True)
        historial = obtener_historial(u["id"])
        if not historial:
            st.info("Aún no tienes predicciones. ¡Clasifica una fruta primero!")
        else:
            for h in historial:
                emoji = EMOJIS_FRUTA.get(h["fruta"], "🍑")
                icono = "✅" if h["estado"] == "Fresca" else "🔴"
                color = FRESH if h["estado"] == "Fresca" else ROTTEN
                fecha = h["fecha"].strftime("%d/%m/%Y %H:%M") if h["fecha"] else "—"
                st.markdown(f"""
                <div class="hist-row">
                    <div style="font-size:1.8rem">{emoji}</div>
                    <div style="flex:1">
                        <div style="font-weight:700;color:{TEXT};font-size:.95rem">{h["fruta"]}</div>
                        <div style="font-size:.72rem;color:{MUTED}">{h["modelo"]} · {fecha}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="color:{color};font-weight:600;font-size:.85rem">{icono} {h["estado"]}</div>
                        <div style="font-size:.7rem;color:{MUTED};font-family:'JetBrains Mono',monospace">
                            {h["confianza_fruta"]*100:.1f}% fruta · {h["confianza_estado"]*100:.1f}% estado
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Tab Estadísticas ──────────────────────────────────────────────────────
    elif tab == "estadisticas":
        st.markdown('<div class="main-title">Estadís<span>ticas</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="main-subtitle">Uso de modelos y frutas más analizadas</div>', unsafe_allow_html=True)
        stats  = obtener_estadisticas()
        frutas = obtener_resumen_frutas()
        total  = sum(s["total_predicciones"] for s in stats)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{total}</div><div class="stat-label">Total predicciones</div></div>', unsafe_allow_html=True)
        with col2:
            top_modelo = stats[0]["modelo"] if stats else "—"
            st.markdown(f'<div class="stat-card"><div class="stat-value" style="font-size:1.4rem">{top_modelo}</div><div class="stat-label">Modelo más usado</div></div>', unsafe_allow_html=True)
        with col3:
            top_fruta = frutas[0]["fruta"] if frutas else "—"
            emoji_tf  = EMOJIS_FRUTA.get(top_fruta, "🍑")
            st.markdown(f'<div class="stat-card"><div class="stat-value" style="font-size:1.4rem">{emoji_tf} {top_fruta}</div><div class="stat-label">Fruta más analizada</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="bar-section-title">Uso por modelo</div>', unsafe_allow_html=True)
            for s in stats:
                pct = (s["total_predicciones"] / total * 100) if total > 0 else 0
                st.markdown(f"""
                <div class="bar-row-c">
                    <span class="bar-name">{s["modelo"]}</span>
                    <div class="bar-track"><div class="bar-fill-a" style="width:{pct:.1f}%"></div></div>
                    <span class="bar-pct">{s["total_predicciones"]}</span>
                </div>
                <div style="font-size:.68rem;color:{MUTED};margin-bottom:10px;padding-left:130px">
                    Confianza promedio: {s["avg_confianza_fruta"] or 0}%
                </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="bar-section-title">Frutas más analizadas</div>', unsafe_allow_html=True)
            total_frutas = sum(f["total"] for f in frutas) or 1
            for f in frutas:
                pct   = f["total"] / total_frutas * 100
                emoji = EMOJIS_FRUTA.get(f["fruta"], "🍑")
                st.markdown(f"""
                <div class="bar-row-c">
                    <span class="bar-name">{emoji} {f["fruta"]}</span>
                    <div class="bar-track"><div class="bar-fill-a" style="width:{pct:.1f}%"></div></div>
                    <span class="bar-pct">{f["total"]}</span>
                </div>""", unsafe_allow_html=True)

        if stats:
            st.markdown("---")
            st.markdown('<div class="bar-section-title">Detalle por modelo</div>', unsafe_allow_html=True)
            df = pd.DataFrame(stats)[["modelo","total_predicciones","total_frescas","total_podridas","avg_confianza_fruta","avg_confianza_estado"]]
            df.columns = ["Modelo","Total","Frescas","Podridas","Conf. Fruta %","Conf. Estado %"]
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Tab Panel Admin ───────────────────────────────────────────────────────
    elif tab == "panel_admin":
        st.markdown('<div class="main-title">Panel <span>Admin</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="main-subtitle">Gestión de usuarios y roles</div>', unsafe_allow_html=True)

        usuarios = obtener_usuarios()
        for usr in usuarios:
            es_admin_actual = usr["id"] == u["id"]
            col1, col2, col3, col4 = st.columns([2.5, 1.5, 1.5, 1])
            with col1:
                st.markdown(
                    f'<div style="padding:8px 0"><strong style="color:{TEXT}">{usr["username"]}</strong>'
                    f'<br><span style="font-size:.72rem;color:{MUTED}">{usr["email"]}</span></div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(f'<div style="padding-top:10px">{badge_rol(usr["rol"])}</div>', unsafe_allow_html=True)
            with col3:
                if not es_admin_actual:
                    nuevo_rol = st.selectbox(
                        "Rol", options=list(ROLES.keys()),
                        index=list(ROLES.keys()).index(usr["rol"]),
                        format_func=lambda r: ROLES[r],
                        key=f"rol_{usr['id']}",
                        label_visibility="collapsed",
                    )
                    if nuevo_rol != usr["rol"]:
                        res = cambiar_rol(usr["id"], nuevo_rol, u["id"])
                        if res["ok"]:
                            st.success("Rol actualizado")
                            st.rerun()
            with col4:
                if not es_admin_actual:
                    activo = usr.get("activo", True)
                    lbl    = "✅ Activo" if activo else "🔴 Inactivo"
                    if st.button(lbl, key=f"toggle_{usr['id']}"):
                        toggle_usuario_activo(usr["id"], not activo, u["id"])
                        st.rerun()
            st.divider()

    # ── Tab Logs ──────────────────────────────────────────────────────────────
    elif tab == "ver_logs":
        st.markdown('<div class="main-title">Logs de <span>Actividad</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="main-subtitle">Últimos 100 eventos del sistema</div>', unsafe_allow_html=True)

        logs = obtener_logs(100)
        if not logs:
            st.info("No hay logs registrados aún.")
        else:
            nivel_filtro = st.selectbox("Filtrar por nivel", ["TODOS", "INFO", "WARNING", "ERROR"])
            for log in logs:
                if nivel_filtro != "TODOS" and log["nivel"] != nivel_filtro:
                    continue
                nivel = log["nivel"] or "INFO"
                cls   = {"WARNING": "log-warn", "ERROR": "log-error"}.get(nivel, "log-info")
                fecha = log["fecha"].strftime("%d/%m %H:%M:%S") if log["fecha"] else "—"
                user  = log["username"] or "sistema"
                st.markdown(
                    f'<div class="log-row {cls}">'
                    f'<span style="color:{MUTED}">{fecha}</span> '
                    f'<strong>[{nivel}]</strong> '
                    f'<span style="color:{ACCENT}">{log["accion"]}</span> — '
                    f'{log["detalle"] or ""} '
                    f'<span style="color:{MUTED};float:right">👤 {user}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.usuario:
    pantalla_app()
elif st.session_state.pantalla == "registro":
    pantalla_registro()
elif st.session_state.pantalla == "verificar":
    pantalla_verificar()
else:
    pantalla_login()
