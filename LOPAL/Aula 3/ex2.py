print ("---- LOJA VIRUAL----")

valor_compra = float(input("QUAL O VALOR DA COMPRA? R$:"))
cliente_vip = int(input("Você é um cliente VIP? (1-SIM/ 0-NAO)"))

if valor_compra >= 150.00 or cliente_vip == 1:
    print("Parabéns! Você ganhou FRETE GRÁTIS.")
else:
    print("O frete custará R$ 20.00.")
