import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

function UploadForm() {
  const [image, setImage] = useState<File | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const navigate = useNavigate();
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;

    console.log("UploadForm mounted");
    if (!sessionStorage.getItem('tabId')) {
      const tabId = crypto.randomUUID();
      sessionStorage.setItem('tabId', tabId);
      window.name = tabId;
    }

    console.log("tabId: ", window.name);

    fetch("http://localhost:8000/api", {
      credentials: 'include',
      headers: {
        "X-Tab-Id": window.name!,
      },
    });
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
      setMessage(null);
    }
  };

  const handleUpload = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('http://localhost:8000/upload-image', {
        method: 'POST',
        body: formData,
        credentials: 'include',
        headers: {
          'X-Tab-Id': window.name!,
        },
      });

      const data = await response.json();

      if (!response.ok) throw new Error(data.detail || 'Upload failed');

      if (data.board && data.original_image) {
        sessionStorage.removeItem('board-data');
        sessionStorage.removeItem('cell-data');
        sessionStorage.removeItem('show-notations');
        navigate('/gamepage', {
          state: {
            board: data.board,
            original_image: data.original_image,
            wrapped_image: data.wrapped_image,
          },
        });
      } else {
        setMessage(`✅ ${data.message}`);
      }
    } catch (error: any) {
      setMessage(`❌ ${error.message}`);
    }
  };

  return (
    <div style={{ padding: '1rem', border: '1px solid #ccc', marginTop: '2rem' }}>
      <h2>Upload Sudoku Image</h2>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginLeft: '1rem' }}>
        Upload
      </button>
      {message && <p style={{ marginTop: '1rem' }}>{message}</p>}
    </div>
  );
}

export default UploadForm;