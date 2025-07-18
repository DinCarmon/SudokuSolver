import { Routes, Route } from 'react-router-dom';
import UploadForm from './UploadForm';
import GamePage from './GamePage';
import './App.css'

function App() {
    /*
        This is an example for a UI generated from the react-ts template.
     */
    /*
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )

     */
    return (
        <Routes>
          <Route path="/uploadform" element={<UploadForm/>}/>
          <Route path="/gamepage" element={<GamePage/>} />
          <Route path="*" element={<UploadForm/>} /> {/* default */}
        </Routes>
      );
}

export default App
