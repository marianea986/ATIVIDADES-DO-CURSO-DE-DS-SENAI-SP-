print ("Bem-vido ao Sistema Escolar!")
nome = input ("Digite o nome do aluno: ")

prova1 = float(input("Nota da prova 1: "))
prova2 = float(input("Nota da prova 2: "))
prova3 = float(input("Nota da prova 3: "))
trabalho = float(input("Nota do trabalho: "))

media_provas = (prova1 + prova2 + prova3 / 3)
media_final = (media_provas + trabalho / 2)
print (f"Sua média final é: {media_final}=.1f ")

if media_final >= 7:
    print("Parabéns você foi aprovado!")
elif media_final < 7:
    print("Você não foi aprovado")
