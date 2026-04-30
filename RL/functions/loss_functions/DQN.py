import torch


def no_replay_loss(
                        gamma: float,
                        model,
                        previous_state,   
                        state,            
                        actions,         
                        r                 
                ):

    # Nous implémentons ici la loss du DQN sans "experience replay" :
    # https://arxiv.org/pdf/1312.5602

    # ''If the weights are updated after every time-step, and the expectations are replaced by 
    # single samples from the behaviour distribution ρ 
    # and the emulator E respectively, then we arrive at the familiar Q-learning algorithm''

    # Cette phrase tirée du papier nous indique :
    # LOSS(theta_{i}) = E[(y_i - Q(s,a;theta_{i}))²] 
    # avec y_i = E[r + gamma* max(over a'){Q(s',a';theta_{i-1})} | s,a] 
    # peuvent être résumés à : 
    # LOSS(theta_i) = mean( (r + gamma*max_a'[ Q(s', a' ; theta_{i-1}) ] - Q(s, a ; theta_i))² )
    
    #avec :
    # Q(s, a ; theta_i)= model(previous_state)
    # Q(s', a'; theta_{i-1}) = model(state).detach()
    #     .detach() car theta_{i-1} est fixé (pas de gradient sur la cible)
    # max_a'[Q(s', a')] = model(state).detach().max(dim=1).values
    # y_i = r + gamma * max_a'[Q(s', a')]                
    #   on prend Q(s, a) pour l'action effectivement jouée
    #     model(previous_state) : (batch_size, 2), on indexe la colonne de l'action jouée via .gather
    # Au final :
    # LOSS = mean( ( r + gamma * max_a'[Q(s', a' ; theta_{i-1})] - Q(s, a ; theta_i) )² )

    with torch.no_grad():
        q_next = model(state)                        
        q_next_max = q_next.max(dim=1).values        
        y = r + gamma * q_next_max                   

    q_all = model(previous_state)                    
    action_idx = actions.long()                      
    q_current = q_all.gather(1, action_idx.unsqueeze(1)).squeeze(1)

    loss = ((y - q_current).pow(2)).mean()
    return loss