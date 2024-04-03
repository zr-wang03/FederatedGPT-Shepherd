Step 1: go to /shakespeare and run preprocess.sh

    sudo bash preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8


Note that the shakespeare full dataset is too big with 4,226,15 samples, we only sample 20% of them to use. 




Step 2: make sure there is data_leaf folder that contains training folder and testing folder, each one should include a user folder. Run load_data.py

This turn original data into pairs like this:

    {
        "instruction": "What letter should be after this:\"om this city; For whom, and not for Tybalt, Juliet pin'd. You, to remove that si\"?",
        "context": "",
        "response": "e",
        "category": "THE_TRAGEDY_OF_ROMEO_AND_JULIET_FRIAR"
    },

where the category contains the name of the user(useful in step 3).



Step 3: Run client_data_allocation with 

    num_client=10 # The number of clients
    diff_quantity=0 # Whether clients have different amounts of data
    python client_data_allocation.py $num_client $diff_quantity



Step 4: Run the program with 

    python main.py --global_model 'chavinlo/alpaca-native'\
        --data_path  "./data" \
        --output_dir  './lora-shepherd-7b/'\
        --num_communication_rounds 10 \
        --num_clients  10 \
        --train_on_inputs \
        --group_by_length