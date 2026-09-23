nome = input("Digite o seu nome: ")
idade = int(input("Digite a sua idade: "))
nota = float(input("Digite a sua nota no teste de lógica: "))
sabe_python = int(input("Sabe Python? (Digite 1 para Sim e 0 para Não): "))

apto = (idade >= 16) and (nota >= 7.0) and (sabe_python == 1)

print(f"\nCandidato: {nome}")
print(f"Apto para a vaga: {apto}")