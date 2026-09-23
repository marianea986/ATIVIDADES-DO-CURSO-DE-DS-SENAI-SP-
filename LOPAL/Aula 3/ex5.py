print ("----SISTEMA DE SEMÁFORO----")
cor = input ("Qual a cor do semáforo? (verde/amarelo/vermelho)")

if cor == "verde":
    print("Siga em frente!")
elif cor == "amarelo":
    print("ATENÇÃO! Reduza a velocidade!")
elif cor == "vermelho":
    print("PARE! você não pode seguir em frente!")
else:
    print("ERRO! As cores são apenas verde/amarelo/vermelho!")