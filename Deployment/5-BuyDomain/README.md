A. First we shall need a permanent ip, so that we will not need to update the dns records all the time

By Default (Dynamic Public IP)
	•	When you start an EC2 instance, AWS assigns it a dynamic public IP address.
	•	This IP:
	•	Changes every time you stop and start the instance.
	•	Does not change when you reboot the instance.
	•	If you stop the instance and start it again, a new public IP will be assigned.
If You Want a Permanent IP (Elastic IP)

To make your public IP permanent, do this:
	1.	Go to the AWS EC2 Console.
	2.	On the left menu, choose “Elastic IPs”.
	3.	Click “Allocate Elastic IP address”.
	4.	After allocation, click “Associate Elastic IP address” and choose your instance.
	5.	Done — now your instance has a static public IP.

1. Go to namecheap and buy a domain. I bought sudokool.space

2. Go to Namecheap Dashboard
	•	Login to Namecheap.
	•	Go to Account → Domain List.
	•	Click “Manage” next to your domain.

⸻

3. Set Namecheap to Use Basic DNS
	•	Under the “Nameservers” section:
	•	Choose “Namecheap BasicDNS” (unless you’re using a custom DNS service like Route53).
	•	Click the green ✅ checkmark to save.

⸻

4. Add DNS Records

Go to the “Advanced DNS” tab and add type A records with host:
- one with host @ (root domain. i.e sudokool.space)
- one with host www (i.e www.sudokool.space)
Under value write the EC2 machine ip.

Wait for DNS Propagation

DNS changes can take a few minutes to a few hours to propagate. Use tools like https://dnschecker.org to monitor propagation.

⸻

Once that’s done, you should be able to:
	•	Visit yourdomain.com → and reach your server.

5. Change harcoded ip to domain name:
 * sudo nano /etc/nginx/sites-available/frontend -> change the domain there. add both domains like that: <server_name sudokool.space www.sudokool.space>;
 * sudo systemctl restart nginx