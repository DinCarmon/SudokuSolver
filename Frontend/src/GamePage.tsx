import { useState, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';

function GamePage() {
  //const isReload = performance.getEntriesByType('navigation')[0]?.type === 'reload';

  /*if (isReload) {
    window.history.replaceState({ ...window.history.state, usr: null }, '', window.location.href);
    console.log("is reload");
  }*/

  const location = useLocation();
  const originalImage = location.state?.original_image as string | undefined;
  const wrappedImage = location.state?.wrapped_image as string | undefined;

  let board: number[][] = [];
  const [currentBoard, setCurrentBoard] = useState<number[][]>(board);
  // Store the history of boards
  const boardHistoryRef = useRef([board]);

  let cellStatus: Record<string, "ok" | "error"> = {};
  const [currentCellStatus, setCurrentCellStatus] = useState<Record<string, "ok" | "error">>(cellStatus);

  const [currentShowMetaData, setCurrentShowMetaData] = useState<boolean | undefined>(undefined);
  const [currentShowNotations, setCurrentShowNotations] = useState<boolean | undefined>(undefined);

  let metadata : number[][] = [];
  const [currentMetadata, setCurrentMetadata] = useState<number[][]>(metadata);

  let cell_notation;
  const [currentCellNotation, setCurrentCellNotation] = useState<string | undefined>(cell_notation);
  
  // Track selected techniques
  const [selectedTechniques, setSelectedTechniques] = useState<Set<number>>(new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]));

  const [lastUsedTechnique, setLastUsedTechnique] = useState<string | undefined>(undefined);
  const [lastStepDescriptionStr, setLastStepDescriptionStr] = useState<string | undefined>(undefined);
  


  //const boardString = currentBoard ? currentBoard.map(row => row.join(' ')).join('\n') : 'No board data';

  //console.log(boardString);
  //console.log("Original image string:", origina lImage?.slice(0, 100));
  //console.log("Wrapped image string:", wrappedImage?.slice(0, 100));

  useEffect(() => {
    //console.log("location state: ", location.state);
    //console.log("window.history.state: ", window.history.state);

    const savedBoard = sessionStorage.getItem(window.name + '-board-data');

    if (savedBoard) {
      const {board} = JSON.parse(savedBoard);
      setCurrentBoard(board);
      console.log("saved board: ", board);
    }
    else {
      board = location.state?.board;
      console.log("Not found board in session storage, using location state");
      if (!board) {
        console.log("No board found in location.state");
      }
      else {
        console.log("board found in location.state: ", board);
      }
      setCurrentBoard(board);
    }

    const savedCellData = sessionStorage.getItem(window.name + '-cell-data');
    if (savedCellData) {
      const {cellStatus} = JSON.parse(savedCellData);
      setCurrentCellStatus(cellStatus)
      //console.log("saved cell status: ", cellStatus);
    }
    else {
      cellStatus = {};
      //console.log("resetting cell status");
      setCurrentCellStatus(cellStatus);
    }

    const savedShowMetaData = sessionStorage.getItem(window.name + '-show-meta-data');
    if (savedShowMetaData) {
      const {showMetaData} = JSON.parse(savedShowMetaData);
      setCurrentShowMetaData(showMetaData);
      console.log("saved show meta data: ", showMetaData);
    }
    else {
      setCurrentShowMetaData(false);
      console.log("setting show meta data to false. no saved show meta data in session storage");
    }

    const savedShowNotations = sessionStorage.getItem(window.name + '-show-notations');
    if (savedShowNotations) {
      const {showNotations} = JSON.parse(savedShowNotations);
      setCurrentShowNotations(showNotations);
      console.log("saved show notations: ", showNotations);
    }
    else {
      setCurrentShowNotations(false);
      console.log("setting show notations to false. no saved show notations in session storage");
    }

    const savedMetadata = sessionStorage.getItem(window.name + '-meta-data');
    if (savedMetadata) {
      const {metadata} = JSON.parse(savedMetadata);
      setCurrentMetadata(metadata);
      console.log("saved metadata: ", metadata);
    }
    else {
      metadata = [];
      setCurrentMetadata(metadata);
      console.log("setting metadata to empty array. no saved metadata in session storage");
    }

    const savedCellNotation = sessionStorage.getItem(window.name + '-cell-notation');
    if (savedCellNotation) {
      const {cellNotation} = JSON.parse(savedCellNotation);
      setCurrentCellNotation(cellNotation);
      console.log("saved cell notation: ", cellNotation);
    }
  }, []);

  useEffect(() => {
    //console.log("updating saved board: ", currentBoard);
    if (currentBoard && currentBoard.length > 0) {
      sessionStorage.setItem(window.name + '-board-data', JSON.stringify({
        board: currentBoard,
      }));
    }
  }, [currentBoard]); // run every time these change

  useEffect(() => {
    //console.log("updating saved cell status: ", currentCellStatus);
    if (currentCellStatus) {
      //console.log("Updating cell statussssssss");
      sessionStorage.setItem(window.name + '-cell-data', JSON.stringify({
        cellStatus: currentCellStatus,
      }));
    }
  }, [currentCellStatus]); // run every time these change

  useEffect(() => {
    if (currentShowMetaData !== undefined) {
      sessionStorage.setItem(window.name + '-show-meta-data', JSON.stringify({
        showMetaData: currentShowMetaData,
      }));
      console.log("updating saved show meta data: ", currentShowMetaData);
    }
  }, [currentShowMetaData]); // run every time these change

  useEffect(() => {
    if (currentShowNotations !== undefined) {
      sessionStorage.setItem(window.name + '-show-notations', JSON.stringify({
        showNotations: currentShowNotations,
      }));
      console.log("updating saved show notations: ", currentShowNotations);
    }
  }, [currentShowNotations]); // run every time these change

  useEffect(() => {
    if (currentCellNotation) {
      sessionStorage.setItem(window.name + '-cell-notation', JSON.stringify({
        cellNotation: currentCellNotation,
      }));
      console.log("updating saved cell notation: ", currentCellNotation);
    }
  }, [currentCellNotation]); // run every time these change

  useEffect(() => {
    if (currentMetadata) {
      sessionStorage.setItem(window.name + '-meta-data', JSON.stringify({
        metadata: currentMetadata,
      }));
      console.log("updating saved metadata: ", currentMetadata);
    }
  }, [currentMetadata]); // run every time these change

  return (
    <div>
      <h2>Sudoku Game. --ID--: {window.name}</h2>
      <div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start', justifyContent: 'center' }}>        {/* Left section (board and buttons) */}
                 {/* Left section (metadata) */}
         <div>
           { currentShowMetaData ? (
             <>
               <h3>Found metadata</h3>
               <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', textAlign: 'left' }}>
                 {currentMetadata.map((meta, index) => (
                   <p key={index} style={{ margin: '0', textAlign: 'left' }}>
                     {index + 1}. {
                     Number(meta[0]) === 1
                       ? "Line of digit " + Number(meta[1]) + " in block (" + (Number(meta[2]) + 1) + "," + (Number(meta[3]) + 1) + ") is in line " + (Number(meta[2]) * 3 + Number(meta[4]) + 1)
                       : Number(meta[0]) === 2
                       ? "Column of digit " + Number(meta[1]) + " in block (" + (Number(meta[2]) + 1) + "," + (Number(meta[3]) + 1) + ") is in column " + (Number(meta[3]) * 3 + Number(meta[4]) + 1)
                       : "Unknown metadata type: " + meta[0]
                     }
                   </p>
                 ))}
               </div>
             </>
           ) : null }
         </div>
        
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
                const isEmpty = value === 0;
                
                // Parse cell notation if available
                try {
                  if (currentCellNotation) {
                    // Check if it's already an array
                    if (Array.isArray(currentCellNotation)) {
                    } else if (typeof currentCellNotation === 'string') {
                      // Try to parse as JSON, but handle potential issues
                      const trimmed = currentCellNotation.trim();
                      if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
                      } else {
                        console.log('Cell notation is not a valid JSON array:', trimmed);
                      }
                    } else {
                      console.log('Unexpected cell notation format:', currentCellNotation);
                    }
                  }
                } catch (e) {
                  console.error('Failed to parse cell notation:', e);
                  console.error('Raw value was:', currentCellNotation);
                  // Don't throw, just continue with empty array
                }
                
                return (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    style={{
                      borderTop: rowIndex % 3 === 0 ? '2px solid black' : '1px solid #333',
                      borderLeft: colIndex % 3 === 0 ? '2px solid black' : '1px solid #333',
                      borderRight: (colIndex + 1) % 3 === 0 ? '2px solid black' : undefined,
                      borderBottom: (rowIndex + 1) % 3 === 0 ? '2px solid black' : undefined,
                      backgroundColor: value === 0 ? 'rgba(246, 247, 242, 20)' : 'rgba(236, 4, 4, 0)',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      fontSize: '18px',
                      fontWeight: 'bold',
                      width: '40px',
                      height: '40px',
                      boxSizing: 'border-box',
                      position: 'relative'
                    }}
                  >
                    <input
                      type="text"
                      maxLength={1}
                      placeholder = {currentShowNotations === true ? currentCellNotation?.[rowIndex]?.[colIndex] : ''}
                      value={value !== 0 ? value : ''}
                      onChange={async (e) => {
                        const val = e.target.value;
                        if (!/^[1-9]?$/.test(val)) return;

                        const newBoard = currentBoard.map((row: number[], rIdx: number) =>
                          row.map((cell: number, cIdx: number) =>
                            rIdx === rowIndex && cIdx === colIndex ? (val === '' ? 0 : parseInt(val)) : cell
                          )
                        );

                        setCurrentBoard(newBoard);
                        boardHistoryRef.current.push(currentBoard.map(row => [...row]));

                        if (val === '') {
                          setCurrentCellStatus((prev) => {
                            const newStatus = { ...prev };
                            delete newStatus[`${rowIndex}-${colIndex}`];
                            return newStatus;
                          });
                        }
                        else {
                          try {
                            const response = await fetch('http://localhost:8000/solve-sudoku', {
                              method: 'POST',
                              headers: {
                                'Content-Type': 'application/json',
                                'X-Tab-Id': window.name!,
                              },
                              credentials: 'include',
                              body: JSON.stringify({
                                board: newBoard,
                                image: originalImage,
                                image_wrap: wrappedImage,
                              }),
                            });
  
                            if (response.status === 400) {
                              setCurrentCellStatus((prev) => ({
                                ...prev,
                                [`${rowIndex}-${colIndex}`]: "error",
                              }));
                            } else {
                              setCurrentCellStatus((prev) => ({
                                ...prev,
                                [`${rowIndex}-${colIndex}`]: "ok",
                              }));
                              const response_update_cell_notation = await fetch('http://localhost:8000/update-cell-notation', {
                                method: 'POST',
                                headers: {
                                  'Content-Type': 'application/json',
                                  'X-Tab-Id': window.name!,
                                },
                                credentials: 'include',
                                body: JSON.stringify({
                                  row: rowIndex,
                                  col: colIndex,
                                  digit: val,
                                }),
                              });
                              const data_update_cell_notation = await response_update_cell_notation.json();
                              setCurrentCellNotation(data_update_cell_notation.cell_notation);
                              console.log("cell notation: ", data_update_cell_notation.cell_notation);

                              const response = await fetch('http://localhost:8000/get-metadata', {
                                method: 'POST',
                                headers: {
                                  'Content-Type': 'application/json',
                                  'X-Tab-Id': window.name!,
                                },
                                credentials: 'include',
                              });
                              const data = await response.json();
                              setCurrentMetadata(data.metadata);
                              console.log("metadata: ", data.metadata);
                            }
                          } catch (err) {
                            console.error("Validation error:", err);
                          }
                        }
                      }}
                      style={{
                        width: '100%',
                        height: '100%',
                        border: 'none',
                        outline: 'none',
                        textAlign: 'center',
                        fontSize: value !== 0 ? '18px' : '10px',
                        fontWeight: 'bold',
                        backgroundColor:
                          currentCellStatus[`${rowIndex}-${colIndex}`] === "error" && value !== 0
                            ? 'red'
                            : value !== 0
                            ? 'green'
                            : 'white',
                        color: value === 0 ? 'black' : 'white',
                        position: 'relative',
                        zIndex: isEmpty ? 1 : 'auto'
                      }}
                    />
                  </div>
                );
              })
            )}
          </div>
          <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <div 
                style={{
                  width: '40px',
                  height: '20px',
                  backgroundColor: currentShowNotations === true ? 'green' : 'red',
                  borderRadius: '10px',
                  position: 'relative',
                  cursor: 'pointer',
                  transition: 'background-color 0.3s'
                }}
                onClick={async () => {
                  setCurrentShowNotations(!(currentShowNotations === true));
                  const response = await fetch('http://localhost:8000/get-cell-notation', {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                      'X-Tab-Id': window.name!,
                    },
                    credentials: 'include',
                  });
                  const data = await response.json();
                  setCurrentCellNotation(data.cell_notation);
                }}
              >
                <div 
                  style={{
                    width: '16px',
                    height: '16px',
                    backgroundColor: 'white',
                    borderRadius: '50%',
                    position: 'absolute',
                    top: '2px',
                    left: currentShowNotations === true ? '22px' : '2px',
                    transition: 'transform 0.3s'
                  }}
                />
              </div>
              Show cell notations
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <div 
                style={{
                  width: '40px',
                  height: '20px',
                  backgroundColor: currentShowMetaData === true ? 'green' : 'red',
                  borderRadius: '10px',
                  position: 'relative',
                  cursor: 'pointer',
                  transition: 'background-color 0.3s'
                }}
                onClick={async () => {
                  setCurrentShowMetaData(!(currentShowMetaData === true));
                  const response = await fetch('http://localhost:8000/get-metadata', {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                      'X-Tab-Id': window.name!,
                    },
                    credentials: 'include',
                  });
                  const data = await response.json();
                  setCurrentMetadata(data.metadata);
                  console.log("metadata: ", data.metadata);
                }}
              >
                <div 
                  style={{
                    width: '16px',
                    height: '16px',
                    backgroundColor: 'white',
                    borderRadius: '50%',
                    position: 'absolute',
                    top: '2px',
                    left: currentShowMetaData === true ? '22px' : '2px',
                    transition: 'transform 0.3s'
                  }}
                />
              </div>
              Show found metadata
            </label>
          </div>
          <div style={{ marginTop: '1rem' }}>
            <h4>Select solving techniques:</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(1)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(1);
                    } else {
                      newSet.delete(1);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Naked Single row / column</label></summary>
                <p>
                 If only one number is left in a column / line, it can be placed in the empty cell.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(2)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(2);
                    } else {
                      newSet.delete(2);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Naked Single block</label></summary>
                <p>
                If only one number is left in a block, it can be placed in the empty cell.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(3)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(3);
                    } else {
                      newSet.delete(3);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Box Line Interaction</label></summary>
                <p>
                The technique eliminates places for a digit based on the digit already placed in the line / column.
                If a digit is found that for it only one place in a given block is available it updates the board accordingly,
                and returns True.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(4)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(4);
                    } else {
                      newSet.delete(4);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Metadata Box Line Interaction</label></summary>
                <p>
                The technique is used to find new metadata.
                The specific metadata it finds is to find a specific row / col in a block where a digit must be.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(5)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(5);
                    } else {
                      newSet.delete(5);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Column Line Interaction</label></summary>
                <p>
                The technique procedure:
                Look at a specific line / column. Look at the missing numbers.
                If only one number is possible, because all other numbers are already in the column, we found it!
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(6)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(6);
                    } else {
                      newSet.delete(6);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Naked Single</label></summary>
                <p>
                The technique looks at the cell notation matrix. if a cell is found with only one option - yay.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(7)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(7);
                    } else {
                      newSet.delete(7);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Naked Double</label></summary>
                <p>
                TODO: Add description.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(8)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(8);
                    } else {
                      newSet.delete(8);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Naked Triple</label></summary>
                <p>
                In a block, where 3 cells share only 3 total different options of numbers, all other cells in the block
                cannot be filled with these numbers.
                This is also true for lines / columns.
                see <a href="https://www.youtube.com/watch?v=Mh8-MICdO6s&ab_channel=LearnSomething">https://www.youtube.com/watch?v=Mh8-MICdO6s&ab_channel=LearnSomething</a> 9:31 - for an example.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(9)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(9);
                    } else {
                      newSet.delete(9);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Metadata from cell notation</label></summary>
                <p style={{ whiteSpace: 'pre', lineHeight: '1.5', fontFamily: 'monospace' }}>
                 Example: Cell notation in block (1,2) should be that 8 is on the first line<br />
                 
                 -------------------------------------------------------------------------------------<br />
                 |    6     **59***  *4789** | **489**  **489**     1    |    3        2     **578** |<br />
                 |    3        2     *4789** |    6        5     **49*** | **478**     1     **78*** |<br />
                 | **158**  **15***  **148** | **23***     7     **23*** | **458**     6        9    |<br />
                 -------------------------------------------------------------------------------------<br />
                 | **189**     6        3    | *2589**  **289**  **259** | **17***     4     **17*** |<br />
                 |    2        4        5    |    1        6        7    |    9        8        3    |<br />
                 |    7     **19***  **189** | *3489**  **489**  **349** |    2        5        6    |<br />
                 -------------------------------------------------------------------------------------<br />
                 |    4        3     **29*** | **259**     1        6    | **58***     7     **258** |<br />
                 | **15***     7     **12*** | **25***     3        8    |    6        9        4    |<br />
                 | **159**     8        6    |    7     **249**  *2459** | **15***     3     **125** |<br />
                 -------------------------------------------------------------------------------------<br />
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(10)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(10);
                    } else {
                      newSet.delete(10);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Update cell notation from metadata</label></summary>
                <p>
                TODO: Add description.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(11)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(11);
                    } else {
                      newSet.delete(11);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Skyscraper</label></summary>
                <p>
                TODO: Add description.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(12)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(12);
                    } else {
                      newSet.delete(12);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Two String Kite</label></summary>
                <p>
                TODO: Add description.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(13)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(13);
                    } else {
                      newSet.delete(13);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> Empty Rectangle</label></summary>
                <p>
                TODO: Add description.
                </p>
              </details>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <details style={{ textAlign: 'left' }}>
                <summary><label><input 
                  type="checkbox" 
                  checked={selectedTechniques.has(14)}
                  onChange={(e) => {
                    const newSet = new Set(selectedTechniques);
                    if (e.target.checked) {
                      newSet.add(14);
                    } else {
                      newSet.delete(14);
                    }
                    setSelectedTechniques(newSet);
                  }}
                /> XY-Wing</label></summary>
                <p>
                TODO: Add description.
                </p>
              </details>
            </div>
          </div>
          <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'center', gap: '1rem' }}>
            <button 
              style={{ padding: '0.5rem 1rem', backgroundColor: '#4CAF50', color: 'white', border: 'none', borderRadius: '4px' }}
              onClick={async () => {
                const response = await fetch('http://localhost:8000/next-step-sudoku-human-solver', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                    'X-Tab-Id': window.name!,
                  },
                  credentials: 'include',
                  body: JSON.stringify({
                    techniques_to_use: Array.from(selectedTechniques).map(t => t.toString())
                  })
                });
                const data = await response.json();
                if (data.success) {
                  setCurrentBoard(data.board);
                  setCurrentCellNotation(data.cell_notation);
                  setLastUsedTechnique(data.last_used_technique);
                  setLastStepDescriptionStr(data.last_step_description_str);
                }
                else {
                  alert("Failed to find next step.");
                }

                const response2 = await fetch('http://localhost:8000/get-metadata', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                    'X-Tab-Id': window.name!,
                  },
                  credentials: 'include',
                });
                const data2 = await response2.json();
                setCurrentMetadata(data2.metadata);
                console.log("metadata: ", data2.metadata);
              }}
            >
              Next Step
            </button>
            <button
              style={{ padding: '0.5rem 1rem', backgroundColor: '#2196F3', color: 'white', border: 'none', borderRadius: '4px' }}
              onClick={async () => {
                try {
                  const response = await fetch('http://localhost:8000/solve-sudoku', {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                      'X-Tab-Id': window.name!,
                    },
                    credentials: 'include',
                    body: JSON.stringify({
                      board: currentBoard,
                      image: originalImage,
                      image_wrap: wrappedImage
                    })
                  });

                  const data = await response.json();

                  if (response.status === 400) {
                    alert((data.detail || "Invalid board.") + "\nReverting if possible.");
                    if (boardHistoryRef.current.length > 1) {
                      const prevBoard = boardHistoryRef.current.pop();
                      if (prevBoard) {
                        setCurrentBoard(prevBoard);
                        // Find changed cells between currentBoard and prevBoard
                        const newcurrentCellStatus = { ...currentCellStatus };
                        currentBoard.forEach((row, rIdx) => {
                          row.forEach((cell, cIdx) => {
                            if (prevBoard && prevBoard[rIdx] && prevBoard[rIdx][cIdx] !== cell) {
                              // Remove error status for changed cells
                              const key = `${rIdx}-${cIdx}`;
                              if (newcurrentCellStatus[key]) {
                                delete newcurrentCellStatus[key];
                              }
                            }
                          });
                        });
                        setCurrentCellStatus(newcurrentCellStatus);
                      }
                    }
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
                  if (prevBoard) {
                    setCurrentBoard(prevBoard);
                    // Find changed cells between currentBoard and prevBoard
                    const newcurrentCellStatus = { ...currentCellStatus };
                    currentBoard.forEach((row, rIdx) => {
                      row.forEach((cell, cIdx) => {
                        if (prevBoard && prevBoard[rIdx] && prevBoard[rIdx][cIdx] !== cell) {
                          // Remove error status for changed cells
                          const key = `${rIdx}-${cIdx}`;
                          if (newcurrentCellStatus[key]) {
                            delete newcurrentCellStatus[key];
                          }
                        }
                      });
                    setCurrentCellStatus(newcurrentCellStatus);
                    });
                  }
                }
              }}
            >
              Revert
            </button>
          </div>
          <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
             <p style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>Last used technique: {lastUsedTechnique}</p>
             <p style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>Last step description: {lastStepDescriptionStr}</p>
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