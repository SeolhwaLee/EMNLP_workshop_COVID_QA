import subprocess
from fastapi import FastAPI

app = FastAPI()

##########################################################
# Launch a command with pipes
# p = subprocess.Popen(['python -m parlai.scripts.interactive -mf ../model/poly/covid7'], shell=True,
#                      stdout=subprocess.PIPE,
#                      stdin=subprocess.PIPE)
p = subprocess.Popen(['python -m parlai.scripts.interactive -mf fine_tuning_model/covid19_scraped_ver6/poly_encoder_covid19'], shell=True,
                     stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE)

while 1:
    line = p.stdout.readline()
    line = line.strip().decode()
    #print(line)
    if line == '[  polyencoder_type: codes ]':
        break
#the model is ready for interactivation
##########################################################
@app.get("/")
def root(question: str = "What is Covid-19?"):
    # Send the question and get the output
    p.stdin.write(bytes(question, 'utf-8'))
    p.stdin.write(bytes("\n", 'utf-8'))
    p.stdin.flush()
    line = p.stdout.readline()
    line = line.strip().decode() # To interpret as text, decode
    if '[Polyencoder]' in line: # Exclude warnings and other messages
        line = line.split('[Polyencoder]:')
    #print(line)
    result=line[-1]
    return {"question": question, "answers": result}