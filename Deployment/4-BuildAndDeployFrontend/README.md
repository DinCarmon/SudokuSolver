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

