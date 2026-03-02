Trong folder co 2 file quan trong 

- snn_sin_pde.py: Chay thuat toan DGM cho phuong trinh sine-gordon, trong do su dung SNN la model prediction 

- ann_sin_pde.py: Chay thuat toan DGM cho phuong trinh sine-gordon, trong do su dung ANN la model prediction 

1) File snn_sin_pde.py

- Input: Bao gom cac parameters luu trong file config.cfg

- Thuc hien: 

    - Lay cac parameters nhu no_of_layer, node_of_layer, T tu config.cfg

    - Import PDE Class chua trong file pde_class.py

    - Import SNN Class chua trong  folder models. Vi du thuat toan chinh trong bai se chua trong file snn_regression_torch_memberance.py

    - Sau khi co tat ca cac buoc tren thuc hien thuat toan DGM de uoc luong nghiem. Trong do bao gom 2 modes

     - mode='train': Training model trong 20 lan (runs=20), moi lan nhu vay se xuat ra 1 model Model Object
    
     - mode = 'deploy': Deploy model tren bang viec du doan va tinh theorical energy


- Output: 
    - Model Object duoc train xong va luu o folder model_output (Neu mode la train)

    - File csv bao gom cac so lieu nhu mean forecast, std, theorical energy

2) pde_class.py

- Day la file chua tat ca cac pde can duoc uoc luong nhu SineGordon,..

- Moi class pde bao gom cac thuoc tinh nhu initial_condition, cal_dynamic_loss, cal_initial_loss. Moi pde se co cach
tinh khac nhau dua vao tung dac diem khac nhau cua cac pde

3) Folder models

- Day la folder chua cac class cua model SNN. Dac biet la file snn_regression_torch_memberance.py

- Ham forward duoc viet tu cau truc dua tren bai bao 'Spiking Neural Networks for nonlinear regression '. 'https://github.com/ahenkes1/HENKES_SNN/blob/main/src/model.py'

- Trong do co ham calculate_acs_macs_ops, nham tinh so acs, macs cua mo hinh snn. Phuong phap tinh la layer nao cua ann thi tinh operation la acs, con layr nao cua snn thi tinh operation la macs

