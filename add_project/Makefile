start:
	python3 PdfProccesorService.py & \
	python3 PdfProducerService.py & \
	python3 consumer.py

present:
	python3 presenter.py

stop:
	killall -9 PdfProducerService.py || true
	killall -9 PdfProccesorService.py || true
	killall -9 consumer.py || true
	killall -9 presenter.py || true