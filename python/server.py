# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
from http.server import HTTPStatus
import subprocess
import time
import cgi
import sys
import json
import os
import multiprocessing
from animateText import *
import uuid

hostName = "localhost"
serverPort = 3030

def childProcess(textIn):
    # print("child process sleeping")
    # time.sleep(10.0)
    # print("child process waking after 10 seconds")
    animateText(textIn)

class MyServer(BaseHTTPRequestHandler):
    
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>animate text endpoint. all safe</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

    def do_POST(self):
        try:
            datalen = int(self.headers['Content-Length'])
            data = self.rfile.read(datalen)
            obj = json.loads(data)
            if self.path.endswith("/animate"):
                # p = subprocess.Popen(['python3', 'animateText.py', obj["input"]],
                #      cwd="./",
                #      stdout=subprocess.PIPE,
                #      stderr=subprocess.STDOUT,
                #      universal_newlines=True)
                # for line in p.stdout:
                #     print(line, end='')
                
        
                # Attempt 2 
                print("Animating object: {}".format(obj))
                P = multiprocessing.Process(target=childProcess, args=(obj["input"],))
                print("Starting Child Process")
                P.start()
                self.send_response(HTTPStatus.OK)
                self.end_headers()

                # Attempt 1 
                # n = os.fork()
                # # child process
                # if(n == 0):
                #     print("child process sleeping")
                #     time.sleep(10.0)
                #     print("child process waking after 10 seconds")
                #     animateText(obj["input"])
                #     webServer.server_close()
                # elif(n >= 0) :
                #     print("parent process")

        except:
            self.send_error(404, "{}".format(sys.exc_info()[0]))
            print(sys.exc_info())


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')    
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")