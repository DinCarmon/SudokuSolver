Step 2: Install System Dependencies & Deploy FastAPI Backend

Your EC2 instance is now running Ubuntu. We’ll:
	1.	Update packages
	2.	Install Python, pip, and virtualenv
	3.	Set up your FastAPI app (code + dependencies)
	4.	Run your FastAPI app (initial version)
	5.	Optional: make sure it’s reachable from your browser

⸻

🔹 A. Update the server

Run:
sudo apt update && sudo apt upgrade -y

B. Install Python and pip

Run:
sudo apt install python3 python3-pip python3-venv -y

C. Upload your FastAPI project

You have 3 options:

Option 1: Clone from Git
git clone https://github.com/yourusername/your-repo.git
cd your-repo

D. Create a virtual environment

Inside your project directory:
python3 -m venv venv
source venv/bin/activate

E. Install Python dependencies

Make sure your project has a requirements.txt. Then run:
pip install -r requirements.txt

F. Run FastAPI (test it first)

Run with Uvicorn:
uvicorn main:app --host 0.0.0.0 --port 8000

G. Is Port 8000 Open in the EC2 Security Group?

By default, AWS blocks most ports. You need to allow inbound access to port 8000:

How to check:
	1.	Go to EC2 console
	2.	Find your instance → click its Security Group
	3.	In the “Inbound rules” tab:
	•	Check if there’s a rule like:
	•	Type: Custom TCP
	•	Port: 8000
	•	Source: 0.0.0.0/0 (open to world) or your IP

If missing:
	•	Click Edit inbound rules
	•	Add:
	•	Type: Custom TCP
	•	Port range: 8000
	•	Source: 0.0.0.0/0 (or restrict to your IP)

H. Test it in your browser

Open:
http://<your-ec2-ip>:8000/docs

You should see a FastAPI response.
