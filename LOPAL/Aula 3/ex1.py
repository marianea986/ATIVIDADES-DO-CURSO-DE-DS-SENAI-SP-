print ("----JOGO DO PAR OU IMPAR----")

numero = int (input("Digite um número inteiro: "))
resto = numero % 2

if resto == 0:
    print(f"o número {numero} é par.")
else:
    print(f"O número {numero} é ímpar.")
