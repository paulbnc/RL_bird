def action_index(act:str):
    if act!="saut" and act!="rien":
        raise NameError
    return 1 if act=="saut" else 0