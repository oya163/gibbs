FROM python:3
RUN cd /home/ && git clone https://github.com/oya163/gibbs.git
WORKDIR /home/gibbs/
RUN pip install -r requirements.txt
CMD [ "python", "./monolingual_segmentation.py", "-t", "-w", "अमेरिकाद्वारा"]
