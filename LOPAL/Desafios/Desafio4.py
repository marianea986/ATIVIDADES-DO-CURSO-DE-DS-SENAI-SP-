altura = float(input("Digite sua altura: "))
liberado = altura >= 1.45 

print ("-------liberação de catraca-------")
print (f"A sua altura é: {altura}m")
print (f"Status da catraca(liberado): {liberado}")

if liberado:
    print ("Acesso permitido! Aproveite o passeio")
else:
    print ("Acesso negado! altura mínima de 1.45m necessária!") 