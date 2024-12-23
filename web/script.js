const size = 15;
let board = Array.from({ length: size }, () => Array(size).fill(null));
let turn = 0;
let moves = [];
let isComputerFirst = false;
let waitingForComputer = false;
let gameTerminated = false;

document.getElementById('reset').addEventListener('click', resetGame);
document.getElementById('firstMove').addEventListener('change', toggleFirstMove);

document.getElementById('undo').addEventListener('click', undoMove);

function initBoard() {
    const container = document.getElementById('board');
    container.innerHTML = '';
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const spot = document.createElement('div');
            spot.style.width = '30px';
            spot.style.height = '30px';
            spot.style.position = 'absolute';
            spot.style.left = `${j * 30}px`;
            spot.style.top = `${i * 30}px`;
            spot.addEventListener('click', () => placePiece(i, j));
            container.appendChild(spot);
        }
    }
}

function setMessage(content) {
    const msg = document.getElementById("msg");
    msg.innerHTML = content;
}

function clearMessage() {
    const msg = document.getElementById("msg");
    msg.innerHTML = "";
}

function placePiece(r, c) {
    if (gameTerminated) return;
    if (board[r][c] !== null || waitingForComputer) return;
    clearMessage();
    board[r][c] = isComputerFirst ? 'O' : 'X';
    moves.push([r, c, turn + 1]);
    turn++;
    drawPiece(r, c, isComputerFirst ? 'white' : 'black', turn);
    waitingForComputer = true;
    setMessage("waiting");
    sendSequence();
}

function drawPiece(r, c, color, order) {
    const container = document.getElementById('board');
    const piece = document.createElement('div');
    piece.className = `piece ${color}`;
    piece.style.left = `${c * 30 + 15}px`;
    piece.style.top = `${r * 30 + 15}px`;

    const orderElement = document.createElement('span');
    orderElement.className = 'order';
    orderElement.innerText = order;
    orderElement.style.color = color === 'white' ? 'black' : 'white'; // 根据棋子颜色设置次序颜色
    piece.appendChild(orderElement);

    container.appendChild(piece);
}

function sendSequence() {
    const seq = [Math.floor(turn / 2) + 1, ...moves.map(move => move.slice(0, 2)).flat()];
    console.log("seq: ", seq)
    const url = new URL(window.location.href);
    fetch('config.json')
        .then(response => response.json())
        .then(config => {
            const port = config.port;
            console.log(`Using port: ${port}`);
            url.port = port;
            url.pathname = '/';
            console.log("url: ", url)
            fetch(url.toString(), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(seq)
            })
                .then(res => {
                    console.log("res: ", res);
                    if (!res.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return res.json();
                })
                .then(({ x, y, debug }) => {
                    console.log("x:", x, "y:", y);
                    console.log("debug:", debug);
                    waitingForComputer = false;
                    if (x >= 0 && y >= 0 && x < 15 && y < 15) {
                        board[x][y] = isComputerFirst ? 'X' : 'O';
                        moves.push([x, y, turn + 1]);
                        turn++;
                        drawPiece(x, y, isComputerFirst ? 'black' : 'white', turn);
                        clearMessage();
                    } else {
                        setMessage(debug);
                        if (y == 0 || y == 2) undoMove(false);
                        if (y) gameTerminated = true;
                    }
                })
                .catch(err => {
                    setMessage(err);
                    console.error(err);
                    waitingForComputer = false;
                });
        })
        .catch(error => console.error('Error loading config:', error));
}

function resetGame() {
    clearMessage();
    board = Array.from({ length: size }, () => Array(size).fill(null));
    turn = 0;
    moves = [];
    waitingForComputer = false;
    gameTerminated = false;
    initBoard();
    if (isComputerFirst) {
        moves.push([-1, -1, 0]);
        sendSequence();
    } else {
        waitingForComputer = false;
    }
}

function toggleFirstMove() {
    isComputerFirst = !isComputerFirst;
    resetGame();
}

function undoMove(fullTurn = true) {
    if (gameTerminated) return;
    if (moves.length < 1 || waitingForComputer) return;
    if (fullTurn) {
        const lastMove = moves.pop();
        const secondLastMove = moves.pop();
        board[lastMove[0]][lastMove[1]] = null;
        board[secondLastMove[0]][secondLastMove[1]] = null;
        turn -= 2;
    } else {
        const lastMove = moves.pop();
        board[lastMove[0]][lastMove[1]] = null;
        turn--;
    }
    waitingForComputer = false;
    initBoard();
    moves.forEach(([r, c, order], index) => {
        if (r < 0 || c < 0) return;
        drawPiece(r, c, (index % 2 === 0) ? (isComputerFirst ? 'white' : 'black') : (isComputerFirst ? 'black' : 'white'), order);
    });
}

initBoard();
if (isComputerFirst) {
    moves.push([-1, -1, 0]);
    sendSequence();
} else {
    waitingForComputer = false;
}
