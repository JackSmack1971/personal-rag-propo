@echo off
REM Create venv, install, and run
python -m venv .venv
call .\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
copy .env.example .env
echo Edit .env with your keys, then run: python app.py
