Step-by-Step Guide to Run FastAPI as a Service (Step 6)

1. Get paths

Based on what you told me:
	•	Project path: ~/SudokuSolver/Backend
	•	Virtual environment path: ~/SudokuSolver/Backend/venv (you can confirm it — see note below)
	•	Main file: main.py

📌 Confirm this:
Is your virtual environment located in ~/SudokuSolver/Backend/venv?
(If not, tell me where.)

⸻

2. Create the systemd service file

Run:

sudo nano /etc/systemd/system/sudokool-backend.service

Paste this (adjust if your venv path is different):

[Unit]
Description=FastAPI Sudoku Solver Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/SudokuSolver/Backend
ExecStart=/home/ubuntu/SudokuSolver/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target

This runs the app on 127.0.0.1:8000, safe for nginx to proxy.

Save and exit (Ctrl+O, Enter, Ctrl+X).

⸻

3. Reload systemd and enable service

sudo systemctl daemon-reload
sudo systemctl enable sudokool-backend
sudo systemctl start sudokool-backend

4. Check if it’s working

sudo systemctl status sudokool-backend

You should see something like Active: active (running).

⸻

5. Test with curl

Run this to check:

curl http://127.0.0.1:8000/docs

You should see HTML for the FastAPI docs page.

6. Go to the EC2 dashboard. stop and start instance. website should still be visible