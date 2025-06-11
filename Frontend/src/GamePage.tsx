import { useState } from 'react';
import { useLocation } from 'react-router-dom';

function GamePage() {
  const location = useLocation();
  const board = location.state?.board;
  const originalImage = location.state?.original_image as string | undefined;

  const boardString = board ? board.map(row => row.join(' ')).join('\n') : 'No board data';

  console.log(boardString);
  console.log("Original image string:", originalImage?.slice(0, 100));

  return (
    <div>
      <h2>Sudoku Game</h2>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(9, 40px)',
          gridTemplateRows: 'repeat(9, 40px)',
          gap: '2px',
          marginTop: '1rem'
        }}
      >
        {board?.flatMap((row, rowIndex) =>
          row.map((cell: any, colIndex: number) => {
              console.log(`row ${rowIndex}, col ${colIndex}, raw:`, cell, 'parsed:', Number(cell));
            const value = Number(cell);
            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                style={{
                  border: '1px solid #333',
                  backgroundColor: value === 10 ? '#f0f0f0' : '#fff',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  fontSize: '18px',
                  fontWeight: 'bold',
                  width: '40px',
                  height: '40px',
                  boxSizing: 'border-box'
                }}
              >
                {value !== 10 ? value : ''}
              </div>
            );
          })
        )}
      </div>
      {originalImage && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Original Image</h3>
          <img src={originalImage} alt="Uploaded Sudoku" style={{ maxWidth: '100%', height: 'auto' }} />
        </div>
      )}
    </div>
  );
}

export default GamePage;