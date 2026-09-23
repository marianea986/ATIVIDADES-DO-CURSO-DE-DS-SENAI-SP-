idade = int(input(f"Digite sua idade: "))
tem_cnh = True

pode_dirigir = (idade>=18) and (tem_cnh == True)
print (f"Pode dirigir? {pode_dirigir}")