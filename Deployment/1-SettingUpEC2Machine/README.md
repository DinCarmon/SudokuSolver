Step 1: Launch an EC2 Instance (Ubuntu Server)

We’ll create a basic server (virtual machine) on AWS to host both your backend and frontend.

⸻

🔹 Go to EC2 Dashboard
	1.	Visit: https://console.aws.amazon.com/ec2
	2.	Make sure you’re in the AWS region closest to your users (e.g., eu-central-1 or us-east-1).
	3.	Click “Launch Instance”

⸻

🔹 Instance Settings

Fill in these:
	•	Name: my-mvp-server or whatever you want
	•	AMI (OS): Choose Ubuntu Server 22.04 LTS
	•	Instance Type: t2.micro or t3.micro (Free Tier eligible)
	•	Key Pair (login):
	•	If you don’t have one yet:
	•	Click “Create new key pair”
	•	Name it (e.g., my-key)
	•	Choose file format: .pem (Linux/macOS)
	•	Download and save it securely (you’ll use it to SSH)

⸻

🔹 Network Settings
	•	Under “Firewall” (security group rules), allow:
	•	✅ SSH (port 22) — for logging into your server
	•	✅ HTTP (port 80) — for web access
	•	✅ HTTPS (port 443) — for secure HTTPS access

⸻

🔹 Storage
	•	Default is fine (8 GiB) unless you need more.

⸻

🔹 Launch!
	•	Click “Launch Instance”
	•	Wait 1–2 mins until the instance is running

⸻

🔹 Get Your Public IP
	1.	Go back to the EC2 Dashboard
	2.	Find your instance in the list
	3.	Copy the Public IPv4 address (you’ll use this to SSH and test access)

⸻

🧪 Quick Test: SSH into Your Server

Open your terminal and run:
chmod 400 /path/to/your-key.pem
ssh -i /path/to/your-key.pem ubuntu@<your-ec2-public-ip>