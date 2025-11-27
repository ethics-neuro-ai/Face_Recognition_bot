# 🤖 Face Recognition Telegram Bot (Historic Project)

**Status:** Inactive (2023-2025)  

This project was a **Telegram bot used for face recognition**, active between 2023 and 2025.  
It was mainly used in events and private groups as a **digital identity ticket system**, allowing organizers to verify participants’ identities automatically.

---

## ✨ Features

### 🔐 Face Enrollment
Users could send a photo to the bot to register their face.  
The bot stored facial embeddings for future identity verification.

### 🧠 Face Recognition
When a user sent a new photo, the bot compared it with stored identities and returned the best match.

### 📝 Identity Management
- Add / update / remove identities  
- List registered users  
- Check who is sending messages  

### ⚙️ Hybrid Architecture
- **Node.js / JavaScript**: bot logic, Telegram integration  
- **Python modules**: face recognition and embedding extraction  

---

## 📷 Screenshots

Bot in action:

![Bot main interface](path/to/image1.png)

![Face recognition example](path/to/image2.png)

---

## 📦 Tech Stack

- Node.js / JavaScript  
- Python 3
- MongoDB
- `face_recognition` or similar library  
- Telegram Bot API  
- Storage: JSON / SQLite (local)  
- Optional: Pillow / ffmpeg for image processing  

---

## 🔒 Privacy Notes
This bot was designed **for events and experimental use only**.  
Face data was stored and **never shared** externally. With consens of people involved with respect of GDPR.

---

## 🧪 Example Commands

/start - Start the bot
/enroll - Register a face
/whois - Check identity
/delete - Remove a registered user
/list - List all enrolled users


---

## 🧠 Notes
- Conceptually, this bot functions like a “digital ticket system” using face recognition.  
- No longer in active use after 2025.  
- Useful as a reference for hybrid Telegram bots combining JS and Python for CV tasks.
