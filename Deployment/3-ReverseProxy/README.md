A reverse proxy is a server that sits in front of one or more backend servers (like application servers, APIs, or databases) and handles incoming client requests on their behalf. It plays a crucial role in performance, security, and scalability.
What it does:

When a user sends a request to a website:
	•	The request first goes to the reverse proxy
	•	The reverse proxy decides where to send it (which backend server)
	•	It forwards the request, gets the response, and sends it back to the user

Why it’s useful

1. Load Balancing
	•	Distributes traffic across multiple backend servers
	•	Prevents any one server from being overloaded
	•	Increases uptime and responsiveness

2. SSL Termination
	•	Handles HTTPS encryption/decryption at the proxy level
	•	Offloads heavy TLS/SSL operations from backend servers
	•	Backend servers can operate over plain HTTP

3. Caching
	•	Stores copies of frequently requested responses
	•	Reduces load on backend servers
	•	Speeds up response time for users

4. Security & Anonymity
	•	Hides backend server details (like IP addresses)
	•	Can block malicious requests (DDoS protection, WAF)
	•	Acts as a barrier — if a backend crashes, users don’t see it

5. Centralized Authentication
	•	Can handle login/authentication before passing to backend
	•	Simplifies access control

6. URL Routing / Path Mapping
	•	Can send requests to different backend services based on path
	•	/api/ → backend API server
	•	/images/ → image server
	•	/app/ → web app server

A reverse proxy is not a must but in many production environments, it’s strongly recommended or eventually becomes necessary for scalability, security, or maintainability reasons.
In our case it is mainly used to connect an https connection to default ports as done in regular websites to the relevant port of the front end / backend

Risks of Skipping a Reverse Proxy

🔒 1. No SSL Termination
	•	Tools like uvicorn, gunicorn, or Node.js servers can serve HTTPS directly — but managing certificates, renewals, and security headers is much easier with a proxy like Nginx or Traefik.
	•	Without SSL, data can be intercepted (not acceptable for login pages, payments, or APIs).

🌩️ 2. No DDoS Protection or Rate Limiting
	•	A proxy can detect and block abusive requests before they hit your app.

📊 3. Harder to Scale Later
	•	Without a reverse proxy, you can’t easily add more servers or route based on URL paths (e.g., yourdomain.com/api → API, yourdomain.com/app → frontend).

🧱 4. Less Flexibility
	•	Want to host both your frontend and backend on the same domain? You’ll want a reverse proxy to route based on paths or subdomains.

How does a reverse proxy communicate with backend servers?

Typically, the reverse proxy and backend servers communicate over HTTP or HTTP-like protocols, usually within the same network, VPC, or even the same machine (e.g., via localhost or Unix sockets).

Step 3: Set Up Nginx Reverse Proxy

We’ll use Nginx to:
	1.	Serve your backend via standard web ports (80 and 443)

⸻

🔹 A. Install Nginx
sudo apt update
sudo apt install nginx -y

Confirm it installed:

nginx -v

B. Allow HTTP/HTTPS Traffic (Port 80/443)

Make sure your EC2 security group allows:
	•	HTTP (80)
	•	HTTPS (443)

If you allowed them earlier during setup, you’re good. Otherwise, go to your instance’s Security Group → Inbound Rules and add those.

C. Create a Reverse Proxy Config for FastAPI

Create a new config file for your app:

sudo nano /etc/nginx/sites-available/fastapi

<details>
<summary>Click to view Nginx configuration</summary>
<pre><code>
server {
    listen 80;
    server_name yourdomain.com;  # We'll use your public IP if you don’t have a domain yet
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
</code></pre>
</details>

D. Test It in Browser

Now visit:
http://<your-ec2-ip>/docs

You should see your FastAPI response.
* Verify it works without specifying a port

E. Disable port 8000 from the security group

Now we can get rid of opening the port 8000 under the security group policy.



* If any problems occur, use the nginx logs to debug:
  * sudo tail -n 50 /var/log/nginx/error.log
  * sudo tail -n 50 /var/log/nginx/access.log
