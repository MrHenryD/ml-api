from config.settings import SERVER_HOST, SERVER_PORT
from server.app import app

if __name__ == '__main__':
	app.run(SERVER_HOST, SERVER_PORT)