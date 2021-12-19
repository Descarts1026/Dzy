
classes =["bird"]
all_files = []
with open("../train_data/valdata.txt","r") as f:
    all_files = f.readlines()
    for file in all_files:
        filename= file.split(" ")[0].split("/")[-1]
        with open("./groundtruths/"+filename+".txt","w",encoding='utf_8') as ff:
            infos = file.split(" ")[1:]
            for info in infos:
                classname = classes[int(info.split(",")[-1].strip())]
                result = " ".join(info.split(",")[:-1])
                ff.write(classname + " " + result+"\n")
