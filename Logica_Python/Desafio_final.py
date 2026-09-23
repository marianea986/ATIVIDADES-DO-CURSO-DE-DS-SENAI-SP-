print("--- CONTROLE DE QUALIDADE ---")

codigo = input("Digite o código da peça: ")
peso = float(input("Peso (g): "))
inspecao_input = int(input("Inspeção visual (1 para Aprovada / 0 para Reprovada): "))

inspecao_aprovada = inspecao_input == 1

peca_ok = (peso >= 95) and (peso <= 105) and inspecao_aprovada

print("\n--- RESULTADO ---")
print(f"Peça: {codigo}")
print(f"Peso: {peso} g")
print(f"Inspeção visual aprovada: {inspecao_aprovada}")
print(f"Peça dentro das especificações: {peca_ok}")