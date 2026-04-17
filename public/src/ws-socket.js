export function createWebSocketSocket(path = "/ws") {
    const handlers = new Map();
    const onceHandlers = new Map();
    const pendingAcks = new Map();
    let ws = null;
    let manuallyClosed = false;
    let nextId = 1;

    const socket = {
        get connected() {
            return ws?.readyState === WebSocket.OPEN;
        },

        connect() {
            if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
                return;
            }

            manuallyClosed = false;
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${protocol}//${window.location.host}${path}`);

            ws.addEventListener("open", () => {
                dispatch("connect");
            });

            ws.addEventListener("message", (event) => {
                let message;
                try {
                    message = JSON.parse(event.data);
                } catch (error) {
                    console.error("Invalid WebSocket message:", event.data, error);
                    return;
                }

                if (message.replyTo) {
                    const callback = pendingAcks.get(message.replyTo);
                    if (callback) {
                        pendingAcks.delete(message.replyTo);
                        callback(message.data);
                    }
                    return;
                }

                if (message.event) {
                    dispatch(message.event, message.data);
                }
            });

            ws.addEventListener("close", () => {
                dispatch("disconnect");
                if (!manuallyClosed) {
                    setTimeout(() => socket.connect(), 1000);
                }
            });

            ws.addEventListener("error", () => {
                dispatch("error", { message: "WebSocket connection error" });
            });
        },

        disconnect() {
            manuallyClosed = true;
            pendingAcks.clear();
            if (ws) {
                ws.close();
            }
        },

        emit(event, data, ack) {
            const send = () => {
                const message = { event, data };
                if (typeof ack === "function") {
                    const id = String(nextId++);
                    message.id = id;
                    pendingAcks.set(id, ack);
                }
                ws.send(JSON.stringify(message));
            };

            if (socket.connected) {
                send();
                return;
            }

            const onConnect = () => send();
            socket.once("connect", onConnect);
            socket.connect();
        },

        on(event, callback) {
            if (!handlers.has(event)) {
                handlers.set(event, new Set());
            }
            handlers.get(event).add(callback);
        },

        once(event, callback) {
            if (!onceHandlers.has(event)) {
                onceHandlers.set(event, new Set());
            }
            onceHandlers.get(event).add(callback);
        },
    };

    function dispatch(event, data) {
        const callbacks = handlers.get(event);
        if (callbacks) {
            callbacks.forEach((callback) => callback(data));
        }

        const callbacksOnce = onceHandlers.get(event);
        if (callbacksOnce) {
            onceHandlers.delete(event);
            callbacksOnce.forEach((callback) => callback(data));
        }
    }

    socket.connect();
    return socket;
}
