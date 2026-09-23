fatia = float(input("Digite a quantidade de fatias de pizza: "))
amigos = int(input("Digite o número de amigos: "))
pizza = fatia / amigos
cachorro = fatia % amigos

sabor = str(input("Digite o sabor da pizza (use apenas letras minúsculas): "))
if sabor == "chocolate" :
 print ("O cachorro não comerá nenhuma fatia, pois ele não pode comer chocolate!")
else :
 print (f"O cachorro comerá {cachorro} fatias!")
print ("-------Pedaços por pessoa-------")
print (f"Quantas fatias inteiras cada amigo vai comer: {pizza}")





