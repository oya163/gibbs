FROM python:3
WORKDIR /home
RUN git clone https://github.com/oya163/gibbs.git && cd gibbs
RUN pip install -r requirements.txt
CMD [ "python", "./monolingual_segmentation.py", "-t", "-w", "प्रहरीलाई"]
