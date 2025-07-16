Step 4: Build and Deploy the React Frontend

I chose to Serve the React frontend on the same EC2 instance (easiest for MVP)
We can always migrate later

🧱 A. Build the React App Locally

1. cd Frontend
2. sudo apt install npm
2. npm install --save-dev typescript
3. npm audit fix
4. npm run build
5. Vite will output a production build in the dist/ directory.
6. mv dist/* ~/react-frontend

C. Configure Nginx to Serve React

1. Create a new Nginx config (or modify the existing one):
        <code>sudo nano /etc/nginx/sites-available/frontend</code>
2. Paste this config
<pre><code>
server {
    listen 80;
    server_name your-ec2-ip;

    root /home/ubuntu/react-frontend;
    index index.html;

    location / {
        try_files $uri /index.html;
    }
}
</code></pre>
Replace your-ec2-ip with the actual public IP or your future domain name.

3. Enable the site:
sudo ln -s /etc/nginx/sites-available/frontend /etc/nginx/sites-enabled/
4. Remove the old FastAPI-only config (if desired):
sudo rm /etc/nginx/sites-enabled/fastapi
sudo rm /etc/nginx/sites-enabled/default
5. Give read + execute permissions to directories and files:
sudo chmod -R o+rX /home/ubuntu/react-frontend

Run

ls -ld /home
ls -ld /home/ubuntu
ls -ld /home/ubuntu/react-frontend

You should see something like:

drwxr-xr-x ...

If any of them show drwx------, that blocks Nginx.

5. Restart Nginx:
sudo systemctl restart nginx

D. Test in Browser

Visit:
http://<your-ec2-ip>
    