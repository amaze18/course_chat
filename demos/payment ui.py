import streamlit as st
import razorpay

# Initialize Razorpay client
client = razorpay.Client(auth=("rzp_test_lMyUOGNlFE6zYn", "zF7jouTzFb2cQAlCsBfGP5OX"))
print("3333333dd")
# Streamlit app title
st.title("Vidyarang Registration Payment")

# Form for user input
name = st.text_input("Name")
email = st.text_input("Email")
amount = st.number_input("Amount", step=0.01, format="%.2f")

# Button to initiate payment
if st.button("Pay Now"):
    # Create order
    order_data = {
        "amount": int(amount * 100),  # Convert amount to paise
        "currency": "INR",
        "receipt": "Vidyarang Registration",
        "payment_capture": 1
    }
    try:
        order = client.order.create(data=order_data)
        order_id = order['id']
        order_amount = order['amount']
        order_currency = order['currency']
        order_receipt = order['receipt']
        order_status = order['status']
        order_created_at = order['created_at']

        # Display order details
        st.write("Order created successfully!")
        st.write("Order ID:", order_id)
        st.write("Amount:", order_amount / 100, order_currency)
        st.write("Receipt:", order_receipt)
        st.write("Status:", order_status)
        st.write("Created At:", order_created_at)
        
        # Display payment button
        st.markdown(f"### Pay ₹{amount}")
        st.write(f"Use the following test card details for payment: Card No - 4111 1111 1111 1111, CVV - 123, Expiry - Any future date")
        payment_button = f'<form action="https://api.razorpay.com/v1/checkout/embedded" method="post"><input type="hidden" name="key_id" value="rzp_test_lMyUOGNlFE6zYn"><input type="hidden" name="order_id" value="{order_id}"><input type="hidden" name="name" value="{name}"><input type="hidden" name="email" value="{email}"><input type="hidden" name="contact" value=""><input type="hidden" name="currency" value="INR"><input type="hidden" name="amount" value="{amount*100}"><input type="hidden" name="callback_url" value=""><input type="hidden" name="cancel_url" value=""><input type="submit" value="Pay Now"></form>'
        st.markdown(payment_button, unsafe_allow_html=True)
    except Exception as e:
        st.error("Error creating order. Please try again later.")
        st.error(str(e))
