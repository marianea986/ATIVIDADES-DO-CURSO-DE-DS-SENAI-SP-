carteira = float(input("Digite o valor em R$ que você possue em sua carteira: "))
dolar = carteira * 5.00
euro = carteira * 5.40

print ("-------conversão de dinheiro-------")
print (f"Total da sua carteira: {carteira}")
print (f"Total em dólar: {dolar: .2f}")
print (f"Total em euro: {euro: .2f}")
