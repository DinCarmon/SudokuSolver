# Project Technology Overview

This project uses several modern web development tools and languages. Here's a brief introduction:

---

## React

A popular JavaScript library for building user interfaces, especially dynamic and interactive web apps. It lets you create reusable components to build your UI efficiently.

---

## JavaScript

JavaScript is a programming language that runs in web browsers and allows web pages to be interactive. It is the foundation of most modern web development, enabling dynamic content like animations, form validations, and API requests.

---

## Node.js and npm

Node.js is a platform that allows JavaScript to run outside the browser, often used for backend servers and development tools.

npm (Node Package Manager) is a tool that comes with Node.js to install and manage JavaScript libraries and tools needed for development.

---

## TypeScript

A version of JavaScript that adds type safety, meaning it helps catch errors early by enforcing rules about data types. It makes your code easier to understand and maintain.

---

## Vite

A fast build tool and development server for frontend projects. It allows instant updates in the browser as you develop, making coding faster and smoother.

---

## Setting up the Frontend

We use the command:

```
npm create vite@latest frontend -- --template react-ts
```

to quickly generate a React project configured with TypeScript using Vite. Here's what each part means:

- `npm create vite@latest`: Runs the latest version of Vite's project setup tool.
- `frontend`: The name of the folder where the new project will be created.
- `--`: Separates npm arguments from arguments passed to the create command.
- `--template react-ts`: Specifies the project template to use, in this case React with TypeScript.


After running the project creation command, you typically continue with:

- `cd frontend`: Changes your terminal into the newly created `frontend` folder where the project was generated.
- `npm install`: Installs all required project dependencies listed in the `package.json` file.
- `npm run dev`: Starts the Vite development server. This lets you see your app in the browser (usually at http://localhost:5173) and updates it live as you make changes.

---

This overview will help you understand the main technologies behind the project.