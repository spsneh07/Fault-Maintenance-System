# ══════════════════════════════════════════════════════════════════ # LSTM — BEARING FAULT DIAGNOSIS # Same data pipeline as RNN (windowed vibration segments) # Key change: LSTM replaces SimpleRNN → solves vanishing gradient # ══════════════════════════════════════════════════════════════════ from tensorflow.keras.layers import LSTM, Bidirectional # ── STANDARD LSTM ───────────────────────────────────────────────── lstm_model=Sequential([ LSTM(units=64,
            return_sequences=True, # pass all timesteps to next layer input_shape=(1024, 1), # (timesteps=1024, features=1) name='lstm_1' ),
        # LSTM vs SimpleRNN parameter count: # SimpleRNN(64): 64+64×64+64=4, 224 params # LSTM(64): 4 × (64+64×64+64)=16, 896 params (4 gates !) # More parameters → more expressive → better at long sequences Dropout(0.3),

        LSTM(units=32,
            return_sequences=False, # only final state → (batch, 32) name='lstm_2' ),

        Dense(16, activation='relu' ),
        Dropout(0.2),
        Dense(4, activation='softmax' )], name='lstm_bearing' ) lstm_model.summary() # ── BIDIRECTIONAL LSTM (Advanced) ───────────────────────────────── # Processes sequence FORWARD and BACKWARD simultaneously # For vibration: knowing future context helps classify current state bilstm_model=Sequential([ Bidirectional(LSTM(32, return_sequences=True), input_shape=(1024, 1)),
        # Concatenates forward hidden state + backward hidden state # Output: (batch, 1024, 64) — 32 forward + 32 backward Dropout(0.3),
        Bidirectional(LSTM(16)), # return_sequences=False (default) Dense(4, activation='softmax' )], name='bilstm_bearing' ) # ── TRAIN LSTM ──────────────────────────────────────────────────── lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        # Lower LR than CNN — LSTM has more parameters, needs finer updates loss='categorical_crossentropy' ,
        metrics=['accuracy' ]) lstm_hist=lstm_model.fit(X_tr, y_tr, # using the (N, 1024, 1) data from CNN section epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[ EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=7)],
        verbose=1) lstm_loss, lstm_acc=lstm_model.evaluate(X_te, y_te, verbose=0) print(f"LSTM Accuracy: {lstm_acc:.4f}" ) # ── GATE VALUE INSPECTION ───────────────────────────────────────── # Build a model that outputs gate values (for visualization) import tensorflow.keras.backend as K # Get LSTM layer weights lstm_layer=lstm_model.get_layer('lstm_1' ) Wk=lstm_layer.get_weights() # Wk[0]: kernel (input weights) shape (1, 64×4) — 4 gates, 64 units each # Wk[1]: recurrent kernel shape (64, 64×4) # Wk[2]: bias shape (64×4, ) or (2×64×4, ) depending on implementation print(f"Input kernel shape:     {Wk[0].shape}" ) print(f"Recurrent kernel shape: {Wk[1].shape}" ) # The 4-gate ordering in Keras LSTM: [input, forget, cell, output]