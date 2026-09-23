print("--- VERIFICAÇÃO ---")

# Entrada de dados
nome = input("Nome do cliente: ")
valor_compra = float(input("Valor da compra: R$ "))
cadastro_input = int(input("Possui cadastro? (1 - Sim / 0 - Não): "))

tem_cadastro = cadastro_input == 1

participa = (valor_compra >= 200.0) and tem_cadastro

print("\n--- RESULTADO ---")
print(f"Cliente: {nome}")
print(f"Valor da compra: R$ {valor_compra:.2f}")
print(f"Participa da promoção? {participa}")