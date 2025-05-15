
r = [5, 10, 15, 20, 25, 30]
p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

file = open('latex_table.txt', 'w+')

for pruning in p:
    for merging in r:
        with open(f'summary/Combination/Pruning{pruning:.1f}/ViT-Combine-Pruning{pruning:.1f}-Merging{merging}/metrics.txt') as f:
            metrics = [0]*5
            for i in range(4):
                line = f.readline()
                line = line.split(':')
                line = line[1]
                line = line.split(' ')
                value1 = float(line[1])
                value2 = float(line[3])
                if i == 3:
                    metrics[i] = f'{value1:.4f} \pm {value2:.3f}'
                else:
                    metrics[i] = f'{value1:.2f} \pm {value2:.3f}'
                
            line = f.readline()
            line = line.split(':')
            line = line[1]
            metrics[4] = line[1:]
            
            file.write(f'ViT-B/16 (Prunned {int(100*pruning)}\%, r = {merging}) & ${metrics[4][:-1]}\%$ & {metrics[3].split(" ")[0]} & ${metrics[1]}$ & ${metrics[2]}$ & ${metrics[0]}$\\\\\hline\n')
        
    file.write('\n')