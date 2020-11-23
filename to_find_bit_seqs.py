minu = -0.848297119140625
maxu = 0.62554931640625
x = [-1.0, 1, 0.62548828, 0.6255188]
bits = 16
bit_seqs = ''
for d in x:
    SAMPLE = ''
    tmin, tmax = minu, maxu
    for _ in range(bits):
        mid = (tmin + tmax) / 2
        if d > mid:
            SAMPLE += '1'
            tmin = mid
        else:
            SAMPLE += '0'
            tmax = mid
    bit_seqs += SAMPLE + ' '
    # bit_seqs+='{0:b} '.format(d)
bit_seqs = bit_seqs.split()
print(bit_seqs)

for bit_seq in bit_seqs:
    dec, power = 0, 1
    for i in range(bits - 1, -1, -1):
        if bit_seq[i] == '1':
            dec += power
        power = power * 2
    print(dec)
