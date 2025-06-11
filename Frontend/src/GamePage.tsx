import { useState, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';

function GamePage() {
  const location = useLocation();
  const board = location.state?.board;
  const originalImage = location.state?.original_image as string | undefined;
  const wrappedImage = location.state?.wrapped_image as string | undefined;

  const [currentBoard, setCurrentBoard] = useState(board);
  // Store the history of boards
  const boardHistoryRef = useRef([board]);

  const boardString = currentBoard ? currentBoard.map(row => row.join(' ')).join('\n') : 'No board data';

  console.log(boardString);
  //console.log("Original image string:", originalImage?.slice(0, 100));
  //console.log("Wrapped image string:", wrappedImage?.slice(0, 100));


  return (
    <div>
      <h2>Sudoku Game</h2>
      <div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start', justifyContent: 'center' }}>        {/* Left section (board and buttons) */}
        <div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(9, 40px)',
              gridTemplateRows: 'repeat(9, 40px)',
              gap: '2px',
              marginTop: '1rem',
              justifyContent: 'center',
            }}
          >
            {currentBoard?.flatMap((row, rowIndex) =>
              row.map((cell: any, colIndex: number) => {
                const value = Number(cell);
                return (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    style={{
                      borderTop: rowIndex % 3 === 0 ? '2px solid black' : '1px solid #333',
                      borderLeft: colIndex % 3 === 0 ? '2px solid black' : '1px solid #333',
                      borderRight: (colIndex + 1) % 3 === 0 ? '2px solid black' : undefined,
                      borderBottom: (rowIndex + 1) % 3 === 0 ? '2px solid black' : undefined,
                      backgroundColor: value === 0 ? '#f0f0f0' : 'rgba(5,0,0,0)',
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
                    <input
                      type="text"
                      maxLength={1}
                      value={value !== 0 ? value : ''}
                      onChange={(e) => {
                        const val = e.target.value;
                        if (!/^[1-9]?$/.test(val)) return;

                        const newBoard = currentBoard.map((row: number[], rIdx: number) =>
                          row.map((cell: number, cIdx: number) =>
                            rIdx === rowIndex && cIdx === colIndex ? (val === '' ? 0 : parseInt(val)) : cell
                          )
                        );
                        setCurrentBoard(newBoard);
                        boardHistoryRef.current.push(currentBoard.map(row => [...row]));
                      }}
                      style={{
                        width: '100%',
                        height: '100%',
                        border: 'none',
                        outline: 'none',
                        textAlign: 'center',
                        fontSize: '18px',
                        fontWeight: 'bold',
                        backgroundColor: value !== 0 ? 'green' : '#f0f0f0',
                        color: 'white',
                      }}
                    />
                  </div>
                );
              })
            )}
          </div>
          <div style={{ marginTop: '1rem' }}>
            <h4>Select solving techniques:</h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
              <details>
                <summary><label><input type="checkbox" /> Naked Singles</label></summary>
                <p style={{ marginLeft: '1.5rem' }}>
                  A cell has only one possible number it can be. This is the most basic and common technique.
                </p>
              </details>

              <details>
                <summary><label><input type="checkbox" /> Hidden Singles</label></summary>
                <p style={{ marginLeft: '1.5rem' }}>
                  A number can only go in one position in a row, column, or box, even though the cell has other candidates.
                </p>
              </details>

              <details>
                <summary><label><input type="checkbox" /> Pointing Pairs</label></summary>
                <p style={{ marginLeft: '1.5rem' }}>
                  A candidate is confined to a single row or column within a box, so it can be removed from the same row or column outside the box.
                </p>
              </details>

              <details>
                <summary><label><input type="checkbox" /> Box-Line Reduction</label></summary>
                <p style={{ marginLeft: '1.5rem' }}>
                  A candidate appears in only one line (row or column) within a box, allowing you to eliminate it from other cells in that line.
                </p>
              </details>

              <details>
                <summary><label><input type="checkbox" /> X-Wing</label></summary>
                <p style={{ marginLeft: '1.5rem' }}>
                  A pattern involving two rows and two columns where a candidate can only go in exactly two positions in each row/column, forming a rectangle.
                </p>
              </details>
            </div>
          </div>
          <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'center', gap: '1rem' }}>
            <button style={{ padding: '0.5rem 1rem', backgroundColor: '#4CAF50', color: 'white', border: 'none', borderRadius: '4px' }} onClick={() => alert('Next Step clicked')}>
              Next Step
            </button>
            <button
              style={{ padding: '0.5rem 1rem', backgroundColor: '#2196F3', color: 'white', border: 'none', borderRadius: '4px' }}
              onClick={async () => {
                try {
                  const response = await fetch('http://localhost:8000/solve-sudoku', {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                      board: currentBoard,
                      image: originalImage,
                      image_wrap: wrappedImage
                    })
                  });

                  const data = await response.json();

                  if (response.status === 400) {
                    alert(data.detail || "Invalid Sudoku board or unsolvable puzzle.");
                    return;
                  }

                  if (!response.ok) {
                    throw new Error(data.detail || 'Failed to solve Sudoku');
                  }

                  if (data.solved_board) {
                    boardHistoryRef.current.push(currentBoard.map(row => [...row]));
                    setCurrentBoard(data.solved_board);
                  }

                } catch (error: any) {
                  alert(error.message);
                }
              }}
            >
              Solve
            </button>
            <button
              style={{ padding: '0.5rem 1rem', backgroundColor: '#f44336', color: 'white', border: 'none', borderRadius: '4px' }}
              onClick={() => {
                if (boardHistoryRef.current.length > 1) {
                  const prevBoard = boardHistoryRef.current.pop();
                  setCurrentBoard(prevBoard);
                }
              }}
            >
              Revert
            </button>
          </div>
        </div>

        {/* Right section (image) */}
        {originalImage && (
          <div>
            <h3>Original Image</h3>
            <img src={originalImage} alt="Uploaded Sudoku" style={{ width: '200px', height: 'auto', border: '1px solid #ccc' }} />
            {wrappedImage && (
              <>
                <h3>Extracted Grid</h3>
                <img src={wrappedImage} alt="Extracted Sudoku Grid" style={{ width: '200px', height: 'auto', border: '1px solid #ccc', marginTop: '1rem' }} />
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default GamePage;