import sys
import json
from rabbit_core.json_sender import JsonSender
from rabbit_core.config_loader import ConfigLoader
from rabbit_core.connection_factory import ConnectionFactory
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver


class PresenterRequestHandler(BaseHTTPRequestHandler):
    json_sender_instance = None

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            print(f"Received data: {data}")

            if isinstance(data, list):
                for item in data:
                    PresenterRequestHandler.json_sender_instance.send_json(item)
            else:
                PresenterRequestHandler.json_sender_instance.send_json(data)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "success", "message": "Data received and sent to RabbitMQ"}
            self.wfile.write(json.dumps(response).encode('utf-8'))

        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON: {e}")
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "error", "message": "Invalid JSON provided"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "error", "message": "An internal server error occurred"}
            self.wfile.write(json.dumps(response).encode('utf-8'))


class Presenter:
    def __init__(self, json_sender: JsonSender, host='localhost', port=8000):
        self.json_sender = json_sender
        self.host = host
        self.port = port
        # Pass the json_sender instance to the handler
        PresenterRequestHandler.json_sender_instance = self.json_sender

    def start(self):
        server_address = (self.host, self.port)
        httpd = HTTPServer(server_address, PresenterRequestHandler)
        print(f"Presenter HTTP server started on http://{self.host}:{self.port}")
        print("Listening for POST requests with JSON payload (Ctrl+C to stop)...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nPresenter server stopping.")
        finally:
            httpd.server_close()
            print("Presenter server stopped.")


if __name__ == '__main__':
    config = ConfigLoader().load()
    factory = ConnectionFactory(config)
    connection = factory.create_connection()
    sender = JsonSender(connection)

    # You can configure host and port here if needed, e.g., from config or environment variables
    presenter_server = Presenter(json_sender=sender, host='0.0.0.0', port=8080)
    presenter_server.start()