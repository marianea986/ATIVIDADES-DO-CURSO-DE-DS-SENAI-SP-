import os
import json
import cv2
import time
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageOps
from tkinter import filedialog, messagebox

# ==============================================================================
# PALETA DE CORES CORPORATIVA & AMBIENTE
# ==============================================================================
ctk.set_appearance_mode("Dark")

COLOR_LOGIN_BG_LEFT     = "#4A0066"
COLOR_LOGIN_BG_RIGHT    = "#FFFFFF"
COLOR_LOGIN_PURPLE_BTN  = "#6B008A"
COLOR_LOGIN_PURPLE_HOV  = "#4D0063"
COLOR_LOGIN_TEXT_DARK   = "#1E102A"
COLOR_LOGIN_TEXT_MUTED  = "#757575"
COLOR_LOGIN_INPUT_BG    = "#FAFAFA"
COLOR_LOGIN_INPUT_BORDER= "#E0E0E0"

COLOR_APP_BG            = "#0B1120"
COLOR_CARD_BG           = "#151F32"
COLOR_CARD_BORDER       = "#23314D"
COLOR_BLUE_PRIMARY      = "#2563EB"
COLOR_TEXT_MAIN         = "#F8FAFC"
COLOR_TEXT_MUTED        = "#94A3B8"
COLOR_DANGER            = "#EF4444"
COLOR_SUCCESS           = "#10B981"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "database.json")
FOTOS_DIR = os.path.join(BASE_DIR, "fotos_cadastros")

if not os.path.exists(FOTOS_DIR):
    os.makedirs(FOTOS_DIR)


# ==============================================================================
# DESENHO VETORIAL DO CAPACETE ESPORTIVO ROXO
# ==============================================================================
def gerar_icone_capacete(tamanho=(50, 50), cor=COLOR_LOGIN_PURPLE_BTN):
    scale = 4
    w, h = tamanho[0] * scale, tamanho[1] * scale
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    shell_poly = [
        (int(w * 0.45), int(h * 0.15)),
        (int(w * 0.72), int(h * 0.23)),
        (int(w * 0.88), int(h * 0.40)),
        (int(w * 0.82), int(h * 0.60)),
        (int(w * 0.66), int(h * 0.75)),
        (int(w * 0.38), int(h * 0.82)),
        (int(w * 0.22), int(h * 0.70)),
        (int(w * 0.18), int(h * 0.58)),
        (int(w * 0.38), int(h * 0.52)),
        (int(w * 0.20), int(h * 0.38)),
        (int(w * 0.28), int(h * 0.24)),
    ]
    draw.polygon(shell_poly, fill="#7B00B4")

    top_white = [
        (int(w * 0.45), int(h * 0.15)),
        (int(w * 0.70), int(h * 0.23)),
        (int(w * 0.58), int(h * 0.33)),
        (int(w * 0.32), int(h * 0.30)),
    ]
    draw.polygon(top_white, fill="#FFFFFF")

    visor_poly = [
        (int(w * 0.20), int(h * 0.38)),
        (int(w * 0.44), int(h * 0.36)),
        (int(w * 0.40), int(h * 0.56)),
        (int(w * 0.18), int(h * 0.56)),
    ]
    draw.polygon(visor_poly, fill="#121212")
    
    draw.polygon([
        (int(w * 0.20), int(h * 0.40)),
        (int(w * 0.26), int(h * 0.39)),
        (int(w * 0.24), int(h * 0.54)),
        (int(w * 0.19), int(h * 0.54)),
    ], fill="#383838")

    side_white1 = [
        (int(w * 0.48), int(h * 0.38)),
        (int(w * 0.72), int(h * 0.34)),
        (int(w * 0.66), int(h * 0.44)),
        (int(w * 0.46), int(h * 0.44)),
    ]
    draw.polygon(side_white1, fill="#FFFFFF")

    side_white2 = [
        (int(w * 0.42), int(h * 0.60)),
        (int(w * 0.76), int(h * 0.52)),
        (int(w * 0.68), int(h * 0.65)),
        (int(w * 0.38), int(h * 0.71)),
    ]
    draw.polygon(side_white2, fill="#FFFFFF")

    accent_purple = [
        (int(w * 0.44), int(h * 0.62)),
        (int(w * 0.72), int(h * 0.55)),
        (int(w * 0.66), int(h * 0.63)),
        (int(w * 0.40), int(h * 0.69)),
    ]
    draw.polygon(accent_purple, fill="#A855F7")

    draw.polygon(shell_poly, outline="#1A0028", width=scale)
    draw.polygon(visor_poly, outline="#000000", width=scale)

    img = img.resize(tamanho, Image.Resampling.LANCZOS)
    return ctk.CTkImage(light_image=img, dark_image=img, size=tamanho)


# ==============================================================================
# MOTOR DE DETECÇÃO FACIAL HÍBRIDO ULTRASSENSÍVEL
# ==============================================================================
class MotorDeteccaoFacialHibrido:
    def __init__(self):
        self.use_mp = False
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.25
            )
            self.use_mp = True
        except Exception:
            self.use_mp = False

        self.cascades = []
        paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        ]
        for p in paths:
            if os.path.exists(p):
                c = cv2.CascadeClassifier(p)
                if not c.empty():
                    self.cascades.append(c)

    def detectar(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h_orig, w_orig = frame_bgr.shape[:2]

        # 1. Detecção via MediaPipe IA (Resolução para Óculos e Contra-luz)
        if self.use_mp:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.mp_face_detection.process(rgb)
                if results.detections:
                    faces = []
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        x = int(bboxC.xmin * w_orig)
                        y = int(bboxC.ymin * h_orig)
                        w = int(bboxC.width * w_orig)
                        h = int(bboxC.height * h_orig)
                        x, y = max(0, x), max(0, y)
                        w = min(w_orig - x, w)
                        h = min(h_orig - y, h)
                        if w > 20 and h > 20:
                            faces.append((x, y, w, h))
                    if len(faces) > 0:
                        return faces
            except Exception:
                pass

        # 2. Processamento OpenCV Haar Cascades com CLAHE e Gamma
        target_w = 640
        factor = target_w / float(w_orig)
        target_h = max(1, int(h_orig * factor))
        frame_proc = cv2.resize(frame_bgr, (target_w, target_h))
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        gamma = 2.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gray_gamma = cv2.LUT(gray, table)
        gray_gamma_clahe = clahe.apply(gray_gamma)

        variacoes = [gray_gamma_clahe, gray_clahe, gray_gamma, gray, cv2.equalizeHist(gray)]

        for img in variacoes:
            for cascade in self.cascades:
                faces = cascade.detectMultiScale(img, scaleFactor=1.06, minNeighbors=3, minSize=(30, 30))
                if len(faces) > 0:
                    res = []
                    for (x, y, w, h) in faces:
                        res.append((int(x / factor), int(y / factor), int(w / factor), int(h / factor)))
                    return res

        # 3. Detector de Tom de Pele (HSV) com Filtro de Proporção Facial Humana
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 25, 60], dtype=np.uint8)
        upper_skin = np.array([20, 240, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (w_orig * h_orig * 0.02):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / h
                if 0.55 <= aspect <= 1.25:
                    cx, cy = x + w / 2, y + h / 2
                    dist_center = abs(cx - w_orig / 2) + abs(cy - h_orig / 2)
                    candidates.append((x, y, w, h, area, dist_center))

        if candidates:
            candidates.sort(key=lambda c: (-c[4], c[5]))
            best = candidates[0]
            return [(best[0], best[1], best[2], best[3])]

        return []


motor_detector = MotorDeteccaoFacialHibrido()


def detectar_faces(frame_bgr):
    return motor_detector.detectar(frame_bgr)


def carregar_imagem_pil(caminho_ou_frame):
    if isinstance(caminho_ou_frame, str):
        if not os.path.exists(caminho_ou_frame):
            return None
        try:
            pil_img = Image.open(caminho_ou_frame).convert("RGB")
            return ImageOps.exif_transpose(pil_img)
        except Exception:
            return None
    elif isinstance(caminho_ou_frame, np.ndarray):
        if len(caminho_ou_frame.shape) == 3 and caminho_ou_frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(caminho_ou_frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return Image.fromarray(caminho_ou_frame)
    return None


def extrair_rosto_crop_memoria(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if len(crop_bgr.shape) == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr.copy()
    
    gray_eq = cv2.equalizeHist(gray)
    pil_rosto = Image.fromarray(gray_eq).resize((120, 120), Image.Resampling.LANCZOS)
    return np.array(pil_rosto)


def extrair_e_guardar_rosto(caminho_ou_frame, caminho_destino):
    pil_img = carregar_imagem_pil(caminho_ou_frame)
    if pil_img is None:
        return False

    frame_np = np.array(pil_img)
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR) if len(frame_np.shape) == 3 else frame_np
    faces = detectar_faces(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad_w, pad_h = int(w * 0.10), int(h * 0.10)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(gray.shape[1], x + w + pad_w), min(gray.shape[0], y + h + pad_h)
        rosto_np = gray[y1:y2, x1:x2]
    else:
        rosto_np = gray

    rosto_eq = cv2.equalizeHist(rosto_np)
    pil_rosto = Image.fromarray(rosto_eq).resize((120, 120), Image.Resampling.LANCZOS)
    pil_rosto.save(caminho_destino, quality=95)
    return True


def extrair_rosto_memoria(caminho_ou_frame):
    pil_img = carregar_imagem_pil(caminho_ou_frame)
    if pil_img is None:
        return None

    frame_np = np.array(pil_img)
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR) if len(frame_np.shape) == 3 else frame_np
    faces = detectar_faces(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad_w, pad_h = int(w * 0.10), int(h * 0.10)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(gray.shape[1], x + w + pad_w), min(gray.shape[0], y + h + pad_h)
        rosto_np = gray[y1:y2, x1:x2]
    else:
        rosto_np = gray

    rosto_eq = cv2.equalizeHist(rosto_np)
    pil_rosto = Image.fromarray(rosto_eq).resize((120, 120), Image.Resampling.LANCZOS)
    return np.array(pil_rosto)


def comparar_rostos_otimizado(rosto_cam_120, caminho_foto_cadastrada):
    rosto_cad_120 = extrair_rosto_memoria(caminho_foto_cadastrada)
    if rosto_cam_120 is None or rosto_cad_120 is None:
        return 0.0

    r1 = cv2.equalizeHist(rosto_cam_120)
    r2 = cv2.equalizeHist(rosto_cad_120)

    h, w = r1.shape
    miolo_cam = r1[6:h-6, 6:w-6]

    res = cv2.matchTemplate(r2, miolo_cam, cv2.TM_CCOEFF_NORMED)
    _, max_val_tpl, _, _ = cv2.minMaxLoc(res)
    score_tpl = max(0.0, float(max_val_tpl))

    hist1 = cv2.calcHist([r1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([r2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    score_hist = max(0.0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

    similaridade = (score_tpl * 0.70) + (score_hist * 0.30)
    return min(1.0, max(0.0, similaridade))


# ==============================================================================
# DESENHO DE RETÍCULOS E MALHA BIOMÉTRICA (5 PONTOS)
# ==============================================================================
def desenhar_hud_camera_externa(frame):
    fh, fw = frame.shape[:2]
    m = 22
    c_len = 35
    color = (255, 100, 220)

    cv2.line(frame, (m, m), (m + c_len, m), color, 2, cv2.LINE_AA)
    cv2.line(frame, (m, m), (m, m + c_len), color, 2, cv2.LINE_AA)

    cv2.line(frame, (fw - m, m), (fw - m - c_len, m), color, 2, cv2.LINE_AA)
    cv2.line(frame, (fw - m, m), (fw - m, m + c_len), color, 2, cv2.LINE_AA)

    cv2.line(frame, (m, fh - m), (m + c_len, fh - m), color, 2, cv2.LINE_AA)
    cv2.line(frame, (m, fh - m), (m, fh - m - c_len), color, 2, cv2.LINE_AA)

    cv2.line(frame, (fw - m, fh - m), (fw - m - c_len, fh - m), color, 2, cv2.LINE_AA)
    cv2.line(frame, (fw - m, fh - m), (fw - m, fh - m - c_len), color, 2, cv2.LINE_AA)


def desenhar_tracking_rosto_estilo_imagem(frame, x, y, w, h, texto_status="desconhecido", cor_bgr=(0, 0, 255)):
    c_len = int(min(w, h) * 0.25)
    
    cv2.line(frame, (x, y), (x + c_len, y), cor_bgr, 3, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x, y + c_len), cor_bgr, 3, cv2.LINE_AA)

    cv2.line(frame, (x + w, y), (x + w - c_len, y), cor_bgr, 3, cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w, y + c_len), cor_bgr, 3, cv2.LINE_AA)

    cv2.line(frame, (x, y + h), (x + c_len, y + h), cor_bgr, 3, cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x, y + h - c_len), cor_bgr, 3, cv2.LINE_AA)

    cv2.line(frame, (x + w, y + h), (x + w - c_len, y + h), cor_bgr, 3, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - c_len), cor_bgr, 3, cv2.LINE_AA)

    # Malha Biométrica de 5 Pontos (Constelação)
    pt_tl = (x + int(w * 0.30), y + int(h * 0.32))
    pt_tr = (x + int(w * 0.70), y + int(h * 0.32))
    pt_c  = (x + int(w * 0.50), y + int(h * 0.52))
    pt_bl = (x + int(w * 0.36), y + int(h * 0.78))
    pt_br = (x + int(w * 0.64), y + int(h * 0.78))

    pontos = [pt_tl, pt_tr, pt_c, pt_bl, pt_br]

    cv2.line(frame, pt_tl, pt_tr, cor_bgr, 1, cv2.LINE_AA)
    cv2.line(frame, pt_tl, pt_c, cor_bgr, 1, cv2.LINE_AA)
    cv2.line(frame, pt_tr, pt_c, cor_bgr, 1, cv2.LINE_AA)

    cv2.line(frame, pt_c, pt_bl, cor_bgr, 1, cv2.LINE_AA)
    cv2.line(frame, pt_c, pt_br, cor_bgr, 1, cv2.LINE_AA)
    cv2.line(frame, pt_bl, pt_br, cor_bgr, 1, cv2.LINE_AA)

    cv2.line(frame, pt_tl, pt_bl, cor_bgr, 1, cv2.LINE_AA)
    cv2.line(frame, pt_tr, pt_br, cor_bgr, 1, cv2.LINE_AA)

    for p in pontos:
        cv2.circle(frame, p, 5, cor_bgr, -1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_text = 0.55
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(texto_status, font, scale_text, thickness)
    
    tx = max(10, x + (w - text_w) // 2)
    ty = max(35, y - 18)

    cv2.putText(frame, texto_status, (tx, ty), font, scale_text, cor_bgr, thickness, cv2.LINE_AA)


# ==============================================================================
# PERSISTÊNCIA DE DADOS (JSON)
# ==============================================================================
def carregar_dados():
    if not os.path.exists(DB_FILE):
        dados_iniciais = {"funcionarios": [], "visitantes": []}
        salvar_dados(dados_iniciais)
        return dados_iniciais
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"funcionarios": [], "visitantes": []}


def salvar_dados(dados):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)


# ==============================================================================
# APLICAÇÃO PRINCIPAL
# ==============================================================================
class StarPlastApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("STAR PLAST — Sistema Integrado de Segurança")
        self.geometry("1100x750")
        self.configure(fg_color=COLOR_LOGIN_BG_LEFT)
        self.resizable(True, True)
        
        try:
            self.state('zoomed')
        except Exception:
            pass

        self.cap = None
        self.criar_tela_login()

    def criar_tela_login(self):
        self.configure(fg_color="#300045")

        self.card_container = ctk.CTkFrame(self, width=880, height=520, corner_radius=20, fg_color="transparent")
        self.card_container.place(relx=0.5, rely=0.5, anchor="center")
        self.card_container.pack_propagate(False)

        self.left_banner = ctk.CTkFrame(self.card_container, corner_radius=0, fg_color=COLOR_LOGIN_BG_LEFT)
        self.left_banner.pack(side="left", fill="both", expand=True)

        left_content = ctk.CTkFrame(self.left_banner, fg_color="transparent")
        left_content.pack(expand=True, padx=35)

        ctk.CTkLabel(
            left_content, text="Bem-vindo!", font=("Segoe UI", 34, "bold"), text_color="#FFFFFF", anchor="w"
        ).pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(
            left_content, text="Esse é o sistema da\nportaria da Star Plast", font=("Segoe UI", 20, "bold"),
            text_color="#E0B0FF", anchor="w", justify="left"
        ).pack(anchor="w", pady=(0, 25))

        ctk.CTkLabel(
            left_content, text="STAR PLAST  |  Sistema de Segurança", font=("Segoe UI", 11, "bold"), text_color="#C084FC", anchor="w"
        ).pack(anchor="w")

        self.right_panel = ctk.CTkFrame(self.card_container, corner_radius=0, fg_color=COLOR_LOGIN_BG_RIGHT)
        self.right_panel.pack(side="right", fill="both", expand=True)

        self.icone_capacete_login = gerar_icone_capacete(tamanho=(60, 60), cor=COLOR_LOGIN_PURPLE_BTN)
        lbl_capacete_login = ctk.CTkLabel(self.right_panel, image=self.icone_capacete_login, text="")
        lbl_capacete_login.place(relx=0.92, rely=0.06, anchor="ne")

        form_box = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        form_box.pack(expand=True, fill="x", padx=45)

        ctk.CTkLabel(
            form_box, text="Login", font=("Segoe UI", 30, "bold"), text_color=COLOR_LOGIN_TEXT_DARK
        ).pack(pady=(0, 25))

        ctk.CTkLabel(
            form_box, text="Username", font=("Segoe UI", 11, "bold"), text_color=COLOR_LOGIN_TEXT_MUTED
        ).pack(anchor="w", pady=(0, 4))

        self.email_entry = ctk.CTkEntry(
            form_box, placeholder_text="", width=320, height=40, corner_radius=8,
            fg_color=COLOR_LOGIN_INPUT_BG, border_color=COLOR_LOGIN_INPUT_BORDER,
            text_color=COLOR_LOGIN_TEXT_DARK, font=("Segoe UI", 12)
        )
        self.email_entry.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(
            form_box, text="Password", font=("Segoe UI", 11, "bold"), text_color=COLOR_LOGIN_TEXT_MUTED
        ).pack(anchor="w", pady=(0, 4))

        self.senha_entry = ctk.CTkEntry(
            form_box, placeholder_text="", show="*", width=320, height=40, corner_radius=8,
            fg_color=COLOR_LOGIN_INPUT_BG, border_color=COLOR_LOGIN_INPUT_BORDER,
            text_color=COLOR_LOGIN_TEXT_DARK, font=("Segoe UI", 12)
        )
        self.senha_entry.pack(fill="x", pady=(0, 6))

        lbl_forgot = ctk.CTkLabel(
            form_box, text="forgot password?", font=("Segoe UI", 10, "bold"), text_color=COLOR_LOGIN_PURPLE_BTN
        )
        lbl_forgot.pack(anchor="e", pady=(0, 20))

        self.lbl_erro_login = ctk.CTkLabel(form_box, text="", font=("Segoe UI", 11), text_color=COLOR_DANGER)
        self.lbl_erro_login.pack(pady=(0, 8))

        btn_login = ctk.CTkButton(
            form_box, text="Login", command=self.autenticar, width=180, height=42, corner_radius=10,
            fg_color=COLOR_LOGIN_PURPLE_BTN, hover_color=COLOR_LOGIN_PURPLE_HOV,
            font=("Segoe UI", 14, "bold"), text_color="#FFFFFF"
        )
        btn_login.pack(pady=(0, 25))

    def autenticar(self):
        email = self.email_entry.get().strip()
        senha = self.senha_entry.get().strip()

        if (email == "contato@starplast.com.br" or email == "") and senha == "Starplast2026@":
            self.lbl_erro_login.configure(text="")
            self.card_container.destroy()
            self.configure(fg_color=COLOR_APP_BG)
            self.criar_painel_principal()
        else:
            self.lbl_erro_login.configure(text="E-mail ou senha incorretos!")

    def criar_painel_principal(self):
        self.frame_main = ctk.CTkFrame(self, fg_color=COLOR_APP_BG)
        self.frame_main.pack(fill="both", expand=True)

        header = ctk.CTkFrame(self.frame_main, height=65, fg_color=COLOR_CARD_BG, corner_radius=0, border_width=1, border_color=COLOR_CARD_BORDER)
        header.pack(fill="x", side="top")

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left", padx=25, pady=12)

        ctk.CTkLabel(title_box, text="STAR PLAST", font=("Segoe UI", 20, "bold"), text_color=COLOR_BLUE_PRIMARY).pack(side="left")
        ctk.CTkLabel(title_box, text=" | Portaria & Gestão de Acessos", font=("Segoe UI", 14), text_color=COLOR_TEXT_MUTED).pack(side="left")
        
        self.icone_capacete_header = gerar_icone_capacete(tamanho=(45, 45), cor=COLOR_LOGIN_PURPLE_BTN)
        lbl_capacete_header = ctk.CTkLabel(header, image=self.icone_capacete_header, text="")
        lbl_capacete_header.pack(side="right", padx=(10, 25), pady=8)

        btn_sair = ctk.CTkButton(
            header, text="Sair", width=90, height=32, fg_color="transparent", border_width=1,
            border_color=COLOR_CARD_BORDER, text_color=COLOR_TEXT_MUTED, hover_color=COLOR_CARD_BORDER, command=self.reiniciar
        )
        btn_sair.pack(side="right", padx=10)

        top_container = ctk.CTkFrame(self.frame_main, fg_color="transparent")
        top_container.pack(fill="x", padx=25, pady=20)

        dados = carregar_dados()

        self.card_func = ctk.CTkFrame(top_container, width=190, height=80, corner_radius=10, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER)
        self.card_func.pack(side="left", padx=(0, 12))
        self.card_func.pack_propagate(False)
        ctk.CTkLabel(self.card_func, text="FUNCIONÁRIOS", font=("Segoe UI", 10, "bold"), text_color=COLOR_TEXT_MUTED).pack(anchor="w", padx=16, pady=(12, 0))
        self.lbl_count_func = ctk.CTkLabel(self.card_func, text=str(len(dados["funcionarios"])), font=("Segoe UI", 24, "bold"), text_color=COLOR_BLUE_PRIMARY)
        self.lbl_count_func.pack(anchor="w", padx=16)

        self.card_vis = ctk.CTkFrame(top_container, width=190, height=80, corner_radius=10, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER)
        self.card_vis.pack(side="left", padx=12)
        self.card_vis.pack_propagate(False)
        ctk.CTkLabel(self.card_vis, text="VISITANTES", font=("Segoe UI", 10, "bold"), text_color=COLOR_TEXT_MUTED).pack(anchor="w", padx=16, pady=(12, 0))
        self.lbl_count_vis = ctk.CTkLabel(self.card_vis, text=str(len(dados["visitantes"])), font=("Segoe UI", 24, "bold"), text_color=COLOR_SUCCESS)
        self.lbl_count_vis.pack(anchor="w", padx=16)

        btn_action_box = ctk.CTkFrame(top_container, fg_color="transparent")
        btn_action_box.pack(side="right")

        ctk.CTkButton(
            btn_action_box, text="+ Novo Funcionário", command=self.abrir_modal_funcionario, height=40,
            fg_color=COLOR_BLUE_PRIMARY, hover_color="#1D4ED8", font=("Segoe UI", 12, "bold"), corner_radius=8
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            btn_action_box, text="+ Novo Visitante", command=self.abrir_modal_visitante, height=40,
            fg_color="#059669", hover_color="#047857", font=("Segoe UI", 12, "bold"), corner_radius=8
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            btn_action_box, text="⚡ Validar Catraca", command=self.abrir_modal_catraca, height=40,
            fg_color=COLOR_LOGIN_PURPLE_BTN, hover_color=COLOR_LOGIN_PURPLE_HOV, font=("Segoe UI", 12, "bold"), corner_radius=8
        ).pack(side="left", padx=6)

        self.tabview = ctk.CTkTabview(
            self.frame_main, fg_color=COLOR_CARD_BG, segmented_button_selected_color=COLOR_BLUE_PRIMARY,
            segmented_button_selected_hover_color="#1D4ED8", corner_radius=12
        )
        self.tabview.pack(fill="both", expand=True, padx=25, pady=(0, 20))

        self.tab_func = self.tabview.add("   Funcionários   ")
        self.tab_vis = self.tabview.add("   Visitantes   ")

        search_box_func = ctk.CTkFrame(self.tab_func, fg_color="transparent")
        search_box_func.pack(fill="x", padx=10, pady=8)
        self.search_func_entry = ctk.CTkEntry(
            search_box_func, placeholder_text="🔍 Buscar funcionário por nome ou CPF...", width=420, height=36,
            fg_color=COLOR_APP_BG, border_color=COLOR_CARD_BORDER, text_color=COLOR_TEXT_MAIN
        )
        self.search_func_entry.pack(side="left")
        self.search_func_entry.bind("<KeyRelease>", lambda event: self.filtrar_logs_func())

        self.scroll_func = ctk.CTkScrollableFrame(self.tab_func, fg_color=COLOR_APP_BG, corner_radius=8)
        self.scroll_func.pack(fill="both", expand=True, padx=10, pady=10)

        search_box_vis = ctk.CTkFrame(self.tab_vis, fg_color="transparent")
        search_box_vis.pack(fill="x", padx=10, pady=8)
        self.search_vis_entry = ctk.CTkEntry(
            search_box_vis, placeholder_text="🔍 Buscar visitante por nome ou CPF...", width=420, height=36,
            fg_color=COLOR_APP_BG, border_color=COLOR_CARD_BORDER, text_color=COLOR_TEXT_MAIN
        )
        self.search_vis_entry.pack(side="left")
        self.search_vis_entry.bind("<KeyRelease>", lambda event: self.filtrar_logs_vis())

        self.scroll_vis = ctk.CTkScrollableFrame(self.tab_vis, fg_color=COLOR_APP_BG, corner_radius=8)
        self.scroll_vis.pack(fill="both", expand=True, padx=10, pady=10)

        self.atualizar_logs()

    def excluir_funcionario(self, index_f):
        dados = carregar_dados()
        if 0 <= index_f < len(dados["funcionarios"]):
            removido = dados["funcionarios"].pop(index_f)
            if removido.get("foto_path") and os.path.exists(removido["foto_path"]):
                try:
                    os.remove(removido["foto_path"])
                except Exception:
                    pass
            salvar_dados(dados)
            self.after(50, self.atualizar_logs)

    def excluir_visitante(self, index_v):
        dados = carregar_dados()
        if 0 <= index_v < len(dados["visitantes"]):
            removido = dados["visitantes"].pop(index_v)
            if removido.get("foto_path") and os.path.exists(removido["foto_path"]):
                try:
                    os.remove(removido["foto_path"])
                except Exception:
                    pass
            salvar_dados(dados)
            self.after(50, self.atualizar_logs)

    def atualizar_logs(self):
        dados = carregar_dados()
        self.lbl_count_func.configure(text=str(len(dados["funcionarios"])))
        self.lbl_count_vis.configure(text=str(len(dados["visitantes"])))

        self.filtrar_logs_func()
        self.filtrar_logs_vis()

    def filtrar_logs_func(self):
        dados = carregar_dados()
        query = getattr(self, "search_func_entry", None)
        filtro = query.get().lower().strip() if query else ""

        for widget in self.scroll_func.winfo_children():
            widget.destroy()

        funcionarios_com_indice = [
            (idx, f) for idx, f in enumerate(dados["funcionarios"]) 
            if filtro in f["nome"].lower() or filtro in f["cpf"]
        ]

        if not funcionarios_com_indice:
            ctk.CTkLabel(self.scroll_func, text="Nenhum registro encontrado.", font=("Segoe UI", 12), text_color=COLOR_TEXT_MUTED).pack(pady=20)
            return

        for idx, f in funcionarios_com_indice:
            card = ctk.CTkFrame(self.scroll_func, fg_color=COLOR_CARD_BG, corner_radius=8, border_width=1, border_color=COLOR_CARD_BORDER)
            card.pack(fill="x", padx=5, pady=5)

            header = ctk.CTkFrame(card, fg_color="transparent")
            header.pack(fill="x", padx=16, pady=(10, 4))

            ctk.CTkLabel(header, text=f["nome"], font=("Segoe UI", 14, "bold"), text_color=COLOR_TEXT_MAIN).pack(side="left")

            right_box = ctk.CTkFrame(header, fg_color="transparent")
            right_box.pack(side="right")

            if f.get('matricula'):
                mat_tag = ctk.CTkFrame(right_box, fg_color=COLOR_APP_BG, corner_radius=4, border_width=1, border_color=COLOR_CARD_BORDER)
                mat_tag.pack(side="left", padx=(0, 10))
                ctk.CTkLabel(mat_tag, text=f" Matrícula: {f['matricula']} ", font=("Segoe UI", 10, "bold"), text_color="#A855F7").pack(padx=6, pady=2)

            btn_editar = ctk.CTkButton(
                right_box, text="Editar", width=70, height=28, fg_color=COLOR_BLUE_PRIMARY, hover_color="#1D4ED8",
                font=("Segoe UI", 11, "bold"), corner_radius=6,
                command=lambda i=idx, f_obj=f: self.abrir_modal_editar_funcionario(i, f_obj)
            )
            btn_editar.pack(side="left", padx=3)

            btn_excluir = ctk.CTkButton(
                right_box, text="Excluir", width=70, height=28, fg_color=COLOR_DANGER, hover_color="#DC2626",
                font=("Segoe UI", 11, "bold"), corner_radius=6,
                command=lambda i=idx: self.excluir_funcionario(i)
            )
            btn_excluir.pack(side="left", padx=3)

            details = ctk.CTkFrame(card, fg_color="transparent")
            details.pack(fill="x", padx=16, pady=(0, 10))

            info_txt = f"CPF: {f['cpf']}   •   Setor: {f.get('setor', 'N/I')}   •   Nascimento: {f.get('nascimento', 'N/I')}"
            if f.get("foto_path") and os.path.exists(f["foto_path"]):
                info_txt += "   •   [Biometria Cadastrada]"

            ctk.CTkLabel(details, text=info_txt, font=("Segoe UI", 11), text_color=COLOR_TEXT_MUTED, justify="left").pack(anchor="w")

    def filtrar_logs_vis(self):
        dados = carregar_dados()
        query = getattr(self, "search_vis_entry", None)
        filtro = query.get().lower().strip() if query else ""

        for widget in self.scroll_vis.winfo_children():
            widget.destroy()

        visitantes_com_indice = [
            (idx, v) for idx, v in enumerate(dados["visitantes"]) 
            if filtro in v["nome"].lower() or filtro in v["cpf"]
        ]

        if not visitantes_com_indice:
            ctk.CTkLabel(self.scroll_vis, text="Nenhum registro encontrado.", font=("Segoe UI", 12), text_color=COLOR_TEXT_MUTED).pack(pady=20)
            return

        for idx, v in visitantes_com_indice:
            card = ctk.CTkFrame(self.scroll_vis, fg_color=COLOR_CARD_BG, corner_radius=8, border_width=1, border_color=COLOR_CARD_BORDER)
            card.pack(fill="x", padx=5, pady=5)

            header = ctk.CTkFrame(card, fg_color="transparent")
            header.pack(fill="x", padx=16, pady=(10, 4))

            ctk.CTkLabel(header, text=v['nome'], font=("Segoe UI", 14, "bold"), text_color=COLOR_TEXT_MAIN).pack(side="left")

            right_box = ctk.CTkFrame(header, fg_color="transparent")
            right_box.pack(side="right")

            if v.get('estacionamento') == "Sim":
                est_tag = ctk.CTkFrame(right_box, fg_color=COLOR_APP_BG, corner_radius=4, border_width=1, border_color=COLOR_CARD_BORDER)
                est_tag.pack(side="left", padx=(0, 10))
                ctk.CTkLabel(est_tag, text=" Estacionamento ", font=("Segoe UI", 10, "bold"), text_color=COLOR_SUCCESS).pack(padx=6, pady=2)

            btn_editar = ctk.CTkButton(
                right_box, text="Editar", width=70, height=28, fg_color="#059669", hover_color="#047857",
                font=("Segoe UI", 11, "bold"), corner_radius=6,
                command=lambda i=idx, v_obj=v: self.abrir_modal_editar_visitante(i, v_obj)
            )
            btn_editar.pack(side="left", padx=3)

            btn_excluir = ctk.CTkButton(
                right_box, text="Excluir", width=70, height=28, fg_color=COLOR_DANGER, hover_color="#DC2626",
                font=("Segoe UI", 11, "bold"), corner_radius=6,
                command=lambda i=idx: self.excluir_visitante(i)
            )
            btn_excluir.pack(side="left", padx=3)

            details = ctk.CTkFrame(card, fg_color="transparent")
            details.pack(fill="x", padx=16, pady=(0, 10))

            info_txt = f"CPF: {v['cpf']}   •   Motivo: {v.get('motivo', 'N/I')}"
            if v.get('estacionamento') == "Sim":
                info_txt += f"   •   Veículo: {v.get('modelo', '-')} ({v.get('cor', '-')}) — Placa: {v.get('placa', '-')}"
            if v.get("foto_path") and os.path.exists(v["foto_path"]):
                info_txt += "   •   [Biometria Cadastrada]"

            ctk.CTkLabel(details, text=info_txt, font=("Segoe UI", 11), text_color=COLOR_TEXT_MUTED, justify="left").pack(anchor="w")

    def reiniciar(self):
        self.frame_main.destroy()
        self.criar_tela_login()

    def abrir_modal_camera(self, callback_foto):
        modal_cam = ctk.CTkToplevel(self)
        modal_cam.title("Captura de Biometria Facial")
        modal_cam.geometry("700x600")
        modal_cam.configure(fg_color=COLOR_APP_BG)
        modal_cam.transient(self)
        modal_cam.grab_set()

        lbl_titulo = ctk.CTkLabel(modal_cam, text="Posicione o Rosto no Centro", font=("Segoe UI", 16, "bold"), text_color=COLOR_TEXT_MAIN)
        lbl_titulo.pack(pady=(15, 5))

        lbl_video = ctk.CTkLabel(modal_cam, text="Iniciando câmara...", fg_color=COLOR_CARD_BG, corner_radius=12)
        lbl_video.pack(fill="both", expand=True, padx=20, pady=10)

        rodape = ctk.CTkFrame(modal_cam, fg_color="transparent")
        rodape.pack(fill="x", padx=20, pady=15)

        self.cap_cam = cv2.VideoCapture(0)
        if not self.cap_cam.isOpened():
            self.cap_cam = cv2.VideoCapture(1)

        cam_ativa = [True]
        ultimo_frame = [None]

        def atualizar_feed():
            if not cam_ativa[0]:
                return
            ret, frame = self.cap_cam.read()
            if ret and frame is not None:
                ultimo_frame[0] = frame.copy()
                faces = detectar_faces(frame)
                frame_draw = frame.copy()

                desenhar_hud_camera_externa(frame_draw)

                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                    desenhar_tracking_rosto_estilo_imagem(frame_draw, x, y, w, h, texto_status="ANALISANDO ROSTO...", cor_bgr=(0, 255, 255))

                rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).resize((640, 420), Image.Resampling.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 420))
                lbl_video.configure(image=ctk_img, text="")
            
            if cam_ativa[0]:
                modal_cam.after(20, atualizar_feed)

        def fechar():
            cam_ativa[0] = False
            if self.cap_cam is not None and self.cap_cam.isOpened():
                self.cap_cam.release()
            modal_cam.destroy()

        modal_cam.protocol("WM_DELETE_WINDOW", fechar)

        def capturar():
            if ultimo_frame[0] is not None:
                temp_filename = os.path.join(FOTOS_DIR, f"temp_{int(time.time()*1000)}.jpg")
                if extrair_e_guardar_rosto(ultimo_frame[0], temp_filename):
                    callback_foto(temp_filename)
                    fechar()
                else:
                    messagebox.showwarning("Aviso", "Nenhum rosto claro detectado no frame atual. Tente novamente.", parent=modal_cam)

        def carregar_arquivo():
            cam_ativa[0] = False
            caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png *.jpeg *.webp")], parent=modal_cam)
            if caminho:
                temp_filename = os.path.join(FOTOS_DIR, f"temp_{int(time.time()*1000)}.jpg")
                if extrair_e_guardar_rosto(caminho, temp_filename):
                    callback_foto(temp_filename)
                    fechar()
                else:
                    messagebox.showwarning("Aviso", "Não foi possível extrair a biometria da imagem selecionada.", parent=modal_cam)
                    cam_ativa[0] = True
                    atualizar_feed()
            else:
                cam_ativa[0] = True
                atualizar_feed()

        ctk.CTkButton(
            rodape, text="📷 Capturar Foto", command=capturar, height=38,
            fg_color=COLOR_SUCCESS, hover_color="#059669", font=("Segoe UI", 12, "bold")
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            rodape, text="📁 Carregar de Arquivo", command=carregar_arquivo, height=38,
            fg_color=COLOR_CARD_BORDER, hover_color="#334155", font=("Segoe UI", 12, "bold")
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            rodape, text="Cancelar", command=fechar, height=38,
            fg_color=COLOR_DANGER, hover_color="#DC2626", font=("Segoe UI", 12, "bold")
        ).pack(side="right", padx=5)

        atualizar_feed()

    def abrir_modal_funcionario(self, index=None, dados_func=None):
        modal = ctk.CTkToplevel(self)
        titulo_txt = "Editar Funcionário" if dados_func else "Novo Funcionário"
        modal.title(titulo_txt)
        modal.geometry("520x620")
        modal.configure(fg_color=COLOR_APP_BG)
        modal.transient(self)
        modal.grab_set()

        ctk.CTkLabel(modal, text=titulo_txt, font=("Segoe UI", 20, "bold"), text_color=COLOR_TEXT_MAIN).pack(pady=(20, 15))

        body = ctk.CTkFrame(modal, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_CARD_BORDER)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        def criar_campo(rotulo, val_def=""):
            box = ctk.CTkFrame(body, fg_color="transparent")
            box.pack(fill="x", padx=20, pady=6)
            ctk.CTkLabel(box, text=rotulo, font=("Segoe UI", 11, "bold"), text_color=COLOR_TEXT_MUTED).pack(anchor="w")
            ent = ctk.CTkEntry(box, height=34, fg_color=COLOR_APP_BG, border_color=COLOR_CARD_BORDER, text_color=COLOR_TEXT_MAIN)
            ent.pack(fill="x", pady=(2, 0))
            if val_def:
                ent.insert(0, val_def)
            return ent

        ent_nome = criar_campo("Nome Completo *", dados_func.get("nome", "") if dados_func else "")
        ent_cpf = criar_campo("CPF *", dados_func.get("cpf", "") if dados_func else "")
        ent_setor = criar_campo("Setor", dados_func.get("setor", "") if dados_func else "")
        ent_nascimento = criar_campo("Data de Nascimento", dados_func.get("nascimento", "") if dados_func else "")
        ent_matricula = criar_campo("Matrícula", dados_func.get("matricula", "") if dados_func else "")

        foto_temp = [dados_func.get("foto_path", "") if dados_func else ""]

        box_foto = ctk.CTkFrame(body, fg_color="transparent")
        box_foto.pack(fill="x", padx=20, pady=10)

        lbl_status_foto = ctk.CTkLabel(
            box_foto, 
            text="✅ Foto Biométrica Anexada" if (foto_temp[0] and os.path.exists(foto_temp[0])) else "⚠️ Nenhuma Foto Cadastrada", 
            font=("Segoe UI", 11, "bold"),
            text_color=COLOR_SUCCESS if (foto_temp[0] and os.path.exists(foto_temp[0])) else COLOR_DANGER
        )
        lbl_status_foto.pack(side="left")

        def ao_capturar_foto(caminho_foto):
            foto_temp[0] = caminho_foto
            lbl_status_foto.configure(text="✅ Foto Biométrica Capturada", text_color=COLOR_SUCCESS)

        btn_foto = ctk.CTkButton(
            box_foto, text="📷 Tirar / Alterar Foto",
            command=lambda: self.abrir_modal_camera(ao_capturar_foto),
            height=32, fg_color=COLOR_LOGIN_PURPLE_BTN, hover_color=COLOR_LOGIN_PURPLE_HOV
        )
        btn_foto.pack(side="right")

        def salvar():
            nome = ent_nome.get().strip()
            cpf = ent_cpf.get().strip()

            if not nome or not cpf:
                messagebox.showerror("Erro de Validação", "Nome e CPF são campos obrigatórios!", parent=modal)
                return

            dados = carregar_dados()

            foto_final = foto_temp[0]
            if foto_final and "temp_" in foto_final and os.path.exists(foto_final):
                novo_nome = os.path.join(FOTOS_DIR, f"func_{cpf}_{int(time.time())}.jpg")
                try:
                    os.rename(foto_final, novo_nome)
                    foto_final = novo_nome
                except Exception:
                    pass

            item = {
                "nome": nome,
                "cpf": cpf,
                "setor": ent_setor.get().strip(),
                "nascimento": ent_nascimento.get().strip(),
                "matricula": ent_matricula.get().strip(),
                "foto_path": foto_final
            }

            if index is not None and 0 <= index < len(dados["funcionarios"]):
                dados["funcionarios"][index] = item
            else:
                dados["funcionarios"].append(item)

            salvar_dados(dados)
            self.atualizar_logs()
            modal.destroy()

        btn_salvar = ctk.CTkButton(
            modal, text="Salvar Cadastrado", command=salvar, height=40,
            fg_color=COLOR_BLUE_PRIMARY, hover_color="#1D4ED8", font=("Segoe UI", 13, "bold"), corner_radius=8
        )
        btn_salvar.pack(fill="x", padx=20, pady=(0, 20))

    def abrir_modal_editar_funcionario(self, index, dados_func):
        self.abrir_modal_funcionario(index=index, dados_func=dados_func)

    def abrir_modal_visitante(self, index=None, dados_vis=None):
        modal = ctk.CTkToplevel(self)
        titulo_txt = "Editar Visitante" if dados_vis else "Novo Visitante"
        modal.title(titulo_txt)
        modal.geometry("520x680")
        modal.configure(fg_color=COLOR_APP_BG)
        modal.transient(self)
        modal.grab_set()

        ctk.CTkLabel(modal, text=titulo_txt, font=("Segoe UI", 20, "bold"), text_color=COLOR_TEXT_MAIN).pack(pady=(20, 15))

        body = ctk.CTkScrollableFrame(modal, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_CARD_BORDER)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        def criar_campo(rotulo, val_def=""):
            box = ctk.CTkFrame(body, fg_color="transparent")
            box.pack(fill="x", padx=15, pady=6)
            ctk.CTkLabel(box, text=rotulo, font=("Segoe UI", 11, "bold"), text_color=COLOR_TEXT_MUTED).pack(anchor="w")
            ent = ctk.CTkEntry(box, height=34, fg_color=COLOR_APP_BG, border_color=COLOR_CARD_BORDER, text_color=COLOR_TEXT_MAIN)
            ent.pack(fill="x", pady=(2, 0))
            if val_def:
                ent.insert(0, val_def)
            return ent

        ent_nome = criar_campo("Nome Completo *", dados_vis.get("nome", "") if dados_vis else "")
        ent_cpf = criar_campo("CPF *", dados_vis.get("cpf", "") if dados_vis else "")
        ent_motivo = criar_campo("Motivo da Visita", dados_vis.get("motivo", "") if dados_vis else "")

        box_est = ctk.CTkFrame(body, fg_color="transparent")
        box_est.pack(fill="x", padx=15, pady=6)
        ctk.CTkLabel(box_est, text="Estacionamento", font=("Segoe UI", 11, "bold"), text_color=COLOR_TEXT_MUTED).pack(anchor="w")
        
        opt_est = ctk.CTkOptionMenu(
            box_est, values=["Não", "Sim"],
            fg_color=COLOR_APP_BG, button_color=COLOR_CARD_BORDER, text_color=COLOR_TEXT_MAIN, height=34
        )
        opt_est.pack(fill="x", pady=(2, 0))
        if dados_vis and dados_vis.get("estacionamento") == "Sim":
            opt_est.set("Sim")

        frame_veiculo = ctk.CTkFrame(body, fg_color=COLOR_APP_BG, corner_radius=8, border_width=1, border_color=COLOR_CARD_BORDER)
        
        ent_modelo = ctk.CTkEntry(frame_veiculo, placeholder_text="Modelo do Veículo", height=32, fg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
        ent_cor = ctk.CTkEntry(frame_veiculo, placeholder_text="Cor", height=32, fg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
        ent_placa = ctk.CTkEntry(frame_veiculo, placeholder_text="Placa", height=32, fg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)

        if dados_vis and dados_vis.get("estacionamento") == "Sim":
            ent_modelo.insert(0, dados_vis.get("modelo", ""))
            ent_cor.insert(0, dados_vis.get("cor", ""))
            ent_placa.insert(0, dados_vis.get("placa", ""))

        def alternar_veiculo(val):
            if val == "Sim":
                frame_veiculo.pack(fill="x", padx=15, pady=8)
                ent_modelo.pack(fill="x", padx=10, pady=4)
                ent_cor.pack(fill="x", padx=10, pady=4)
                ent_placa.pack(fill="x", padx=10, pady=(4, 10))
            else:
                frame_veiculo.pack_forget()

        opt_est.configure(command=alternar_veiculo)
        alternar_veiculo(opt_est.get())

        foto_temp = [dados_vis.get("foto_path", "") if dados_vis else ""]

        box_foto = ctk.CTkFrame(body, fg_color="transparent")
        box_foto.pack(fill="x", padx=15, pady=10)

        lbl_status_foto = ctk.CTkLabel(
            box_foto, 
            text="✅ Foto Biométrica Anexada" if (foto_temp[0] and os.path.exists(foto_temp[0])) else "⚠️ Nenhuma Foto Cadastrada", 
            font=("Segoe UI", 11, "bold"),
            text_color=COLOR_SUCCESS if (foto_temp[0] and os.path.exists(foto_temp[0])) else COLOR_DANGER
        )
        lbl_status_foto.pack(side="left")

        def ao_capturar_foto(caminho_foto):
            foto_temp[0] = caminho_foto
            lbl_status_foto.configure(text="✅ Foto Biométrica Capturada", text_color=COLOR_SUCCESS)

        btn_foto = ctk.CTkButton(
            box_foto, text="📷 Tirar / Alterar Foto",
            command=lambda: self.abrir_modal_camera(ao_capturar_foto),
            height=32, fg_color=COLOR_LOGIN_PURPLE_BTN, hover_color=COLOR_LOGIN_PURPLE_HOV
        )
        btn_foto.pack(side="right")

        def salvar():
            nome = ent_nome.get().strip()
            cpf = ent_cpf.get().strip()

            if not nome or not cpf:
                messagebox.showerror("Erro de Validação", "Nome e CPF são campos obrigatórios!", parent=modal)
                return

            dados = carregar_dados()

            foto_final = foto_temp[0]
            if foto_final and "temp_" in foto_final and os.path.exists(foto_final):
                novo_nome = os.path.join(FOTOS_DIR, f"vis_{cpf}_{int(time.time())}.jpg")
                try:
                    os.rename(foto_final, novo_nome)
                    foto_final = novo_nome
                except Exception:
                    pass

            item = {
                "nome": nome,
                "cpf": cpf,
                "motivo": ent_motivo.get().strip(),
                "estacionamento": opt_est.get(),
                "modelo": ent_modelo.get().strip() if opt_est.get() == "Sim" else "",
                "cor": ent_cor.get().strip() if opt_est.get() == "Sim" else "",
                "placa": ent_placa.get().strip() if opt_est.get() == "Sim" else "",
                "foto_path": foto_final
            }

            if index is not None and 0 <= index < len(dados["visitantes"]):
                dados["visitantes"][index] = item
            else:
                dados["visitantes"].append(item)

            salvar_dados(dados)
            self.atualizar_logs()
            modal.destroy()

        btn_salvar = ctk.CTkButton(
            modal, text="Salvar Cadastrado", command=salvar, height=40,
            fg_color="#059669", hover_color="#047857", font=("Segoe UI", 13, "bold"), corner_radius=8
        )
        btn_salvar.pack(fill="x", padx=20, pady=(0, 20))

    def abrir_modal_editar_visitante(self, index, dados_vis):
        self.abrir_modal_visitante(index=index, dados_vis=dados_vis)

    # --------------------------------------------------------------------------
    # VALIDAÇÃO DE CATRACA COM REDE NEURAL E RECONHECIMENTO AUTOMÁTICO
    # --------------------------------------------------------------------------
    def abrir_modal_catraca(self):
        modal_catraca = ctk.CTkToplevel(self)
        modal_catraca.title("STAR PLAST — Validação de Acesso em Catraca")
        modal_catraca.geometry("980x700")
        modal_catraca.configure(fg_color=COLOR_APP_BG)
        modal_catraca.transient(self)
        modal_catraca.grab_set()

        left_panel = ctk.CTkFrame(modal_catraca, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_CARD_BORDER)
        left_panel.pack(side="left", fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(left_panel, text="🎥 Câmera de Monitoramento Ao Vivo", font=("Segoe UI", 15, "bold"), text_color=COLOR_TEXT_MAIN).pack(pady=(12, 6))

        lbl_video_catraca = ctk.CTkLabel(left_panel, text="Conectando à câmara...", fg_color=COLOR_APP_BG, corner_radius=8)
        lbl_video_catraca.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        right_panel = ctk.CTkFrame(modal_catraca, width=320, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_CARD_BORDER)
        right_panel.pack(side="right", fill="y", padx=(0, 15), pady=15)
        right_panel.pack_propagate(False)

        ctk.CTkLabel(right_panel, text="Status da Catraca", font=("Segoe UI", 15, "bold"), text_color=COLOR_TEXT_MAIN).pack(pady=(15, 10))

        card_status = ctk.CTkFrame(right_panel, fg_color=COLOR_APP_BG, corner_radius=8, border_width=1, border_color=COLOR_CARD_BORDER)
        card_status.pack(fill="x", padx=15, pady=5)

        lbl_status_acesso = ctk.CTkLabel(card_status, text="AGUARDANDO APROXIMAÇÃO", font=("Segoe UI", 11, "bold"), text_color=COLOR_TEXT_MUTED)
        lbl_status_acesso.pack(pady=15, padx=10)

        ctk.CTkLabel(right_panel, text="Histórico Recente", font=("Segoe UI", 12, "bold"), text_color=COLOR_TEXT_MUTED).pack(anchor="w", padx=15, pady=(15, 5))

        scroll_historico = ctk.CTkScrollableFrame(right_panel, fg_color=COLOR_APP_BG, corner_radius=8)
        scroll_historico.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        cap_catraca = cv2.VideoCapture(0)
        if not cap_catraca.isOpened():
            cap_catraca = cv2.VideoCapture(1)

        rodando = [True]

        def loop_catraca():
            if not rodando[0]:
                return

            ret, frame = cap_catraca.read()
            if ret and frame is not None:
                faces = detectar_faces(frame)
                frame_draw = frame.copy()

                desenhar_hud_camera_externa(frame_draw)

                dados = carregar_dados()
                todos_cadastros = []
                for f in dados["funcionarios"]:
                    todos_cadastros.append({**f, "tipo": "Funcionário"})
                for v in dados["visitantes"]:
                    todos_cadastros.append({**v, "tipo": "Visitante"})

                melhor_match = None
                maior_sim = 0.0

                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                    crop = frame[y:y+h, x:x+w]
                    rosto_crop_120 = extrair_rosto_crop_memoria(crop)

                    if rosto_crop_120 is not None:
                        for cad in todos_cadastros:
                            foto_path = cad.get("foto_path")
                            if foto_path and os.path.exists(foto_path):
                                sim = comparar_rostos_otimizado(rosto_crop_120, foto_path)
                                if sim > maior_sim:
                                    maior_sim = sim
                                    melhor_match = cad

                    if maior_sim >= 0.45 and melhor_match:
                        nome = melhor_match["nome"]
                        tipo = melhor_match["tipo"]
                        
                        # VERDE: Rosto Cadastrado e Confirmado
                        desenhar_tracking_rosto_estilo_imagem(
                            frame_draw, x, y, w, h, 
                            texto_status="ROSTO CONFIRMADO - CATRACA LIBERADA", 
                            cor_bgr=(0, 255, 128)
                        )
                        lbl_status_acesso.configure(
                            text=f"✅ ACESSO LIBERADO\n{nome} ({tipo})\nSimetria: {int(maior_sim*100)}%",
                            text_color=COLOR_SUCCESS
                        )
                    else:
                        # VERMELHO: Rosto Detectado Não Cadastrado ("desconhecido")
                        desenhar_tracking_rosto_estilo_imagem(
                            frame_draw, x, y, w, h, 
                            texto_status="desconhecido", 
                            cor_bgr=(0, 0, 255)
                        )
                        lbl_status_acesso.configure(
                            text=f"❌ ACESSO NEGADO\nPessoa Não Cadastrada\nSimetria Max: {int(maior_sim*100)}%",
                            text_color=COLOR_DANGER
                        )
                else:
                    lbl_status_acesso.configure(text="AGUARDANDO APROXIMAÇÃO", text_color=COLOR_TEXT_MUTED)

                rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).resize((580, 440), Image.Resampling.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(580, 440))
                lbl_video_catraca.configure(image=ctk_img, text="")

            if rodando[0]:
                modal_catraca.after(30, loop_catraca)

        def fechar():
            rodando[0] = False
            if cap_catraca.isOpened():
                cap_catraca.release()
            modal_catraca.destroy()

        modal_catraca.protocol("WM_DELETE_WINDOW", fechar)
        loop_catraca()


if __name__ == "__main__":
    app = StarPlastApp()
    app.mainloop()