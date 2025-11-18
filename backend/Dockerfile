FROM python:3.10

# set working directory
WORKDIR /app

# copy project
COPY . /app

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# expose port (HF uses 7860 by default, but FastAPI can run on anything)
EXPOSE 7860

# run your start script (HF runs CMD)
CMD ["bash", "start.sh"]
