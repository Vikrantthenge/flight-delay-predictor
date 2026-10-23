    hourly_delay_df,
    x="Hour",
    y="Avg Delay (min)",
    title="Delay Distribution by Departure Hour",
    markers=True,
    line_shape="spline",
    color_discrete_sequence=["#8B0000"]  # Optional: pick a single tone from your reds
)
st.plotly_chart(fig_line, use_container_width=True)

# --- Feature Importance ---
st.subheader("📌 Feature Importance")
st.markdown("""
This chart shows the top 10 feature-importance scores exposed by the loaded model.  
Importance indicates how the model uses features; it does not prove that a feature causes flight delays.
""")
importance_df = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})
 
fig_imp = px.bar(
    importance_df.head(10),
    x="Feature",
    y="Importance",
    title="Top 10 Feature Importances",
    color_discrete_sequence=[custom_reds[3]]
)
st.plotly_chart(fig_imp, use_container_width=True)

# --- SHAP Explainability ---
st.subheader("🧠 SHAP Explainability")
st.markdown("""
SHAP (SHapley Additive exPlanations) helps interpret how features contribute to model predictions.  
The summary below uses five example inputs; it does not explain the current user's prediction.
""")

# Create sample inputs
sample_inputs = pd.concat([
    create_input_df(9, 11, 5, 60, 40, "Indigo", "DEL", "BOM"),
    create_input_df(14, 16, 8, 70, 20, "Spicejet", "MAA", "HYD"),
    create_input_df(6, 8, 3, 50, 60, "Air India", "CCU", "TRV"),
    create_input_df(18, 20, 9, 85, 10, "Akasa Air", "HYD", "DEL"),
    create_input_df(22, 1, 2, 40, 90, "Vistara", "TRV", "MAA")
])

# Generate SHAP values. Compatibility can vary by model and SHAP version.
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_inputs)

    # Select the positive/delayed class for common binary classifier outputs.
    if isinstance(shap_values, list):
        plot_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        plot_values = shap_values
        if getattr(plot_values, "ndim", 0) == 3:
            plot_values = plot_values[:, :, -1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        plot_values,
        sample_inputs,
        plot_type="bar",
        show=False
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    st.image(buf)
except Exception as exc:
    st.warning(
        "SHAP could not generate a plot for the loaded model. "
        f"Check model compatibility and SHAP version. Details: {exc}"
    )

# Layman-friendly explanation
with st.expander("🔍 What does this SHAP plot mean?"):
    st.markdown("""
    - **Longer bars** = Greater average absolute SHAP impact across the example inputs.
    - This bar plot shows impact magnitude, not whether a feature increases or decreases delay risk.
    - Only five example rows are used, so the ranking may be unstable.
    - SHAP explains model behavior; it does not establish the true causes of flight delays.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by **Vikrant Thenge** |")
st.caption("🕒 Last updated: September 2025")
st.markdown("---")

st.markdown("""
<div style='text-align: center; font-size: 16px;'>
    <p><strong style='color: darkred;'>Crafted with precision. Powered by data. Designed for impact.</strong></p>
    <p>
        <a href='https://github.com/vikrantthenge' target='_blank'>
            <img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' width='20' style='vertical-align:middle;' />
            <span style='margin-left:8px;'>GitHub</span>
        </a> &nbsp;&nbsp;&nbsp;
        <a href='https://linkedin.com/in/vthenge' target='_blank'>
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='20' style='vertical-align:middle;' />
            <span style='margin-left:8px;'>LinkedIn</span>
        </a> &nbsp;&nbsp;&nbsp;
        <a href='mailto:vikrantthenge@outlook.com'>
            <img src='https://cdn-icons-png.flaticon.com/512/732/732223.png' width='20' style='vertical-align:middle;' />
            <span style='margin-left:8px;'>Outlook</span>
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
