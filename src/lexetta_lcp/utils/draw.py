import matplotlib.pyplot as plt
from IPython.display import HTML, display

def get_color(score: float) -> str:
    """Convert 0-1 score to green-yellow-red gradient."""
    # score 0 = green, 0.5 = yellow, 1 = red
    if score < 0.5:
        r = int(255 * (score * 2))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (score - 0.5) * 2))
    return f"rgb({r}, {g}, 0)"

def visualize_complexity(sentence: str, word_scores: dict[str, float]) -> None:
    """
    Visualize word complexity in a sentence.
    
    Args:
        sentence: The input sentence
        word_scores: Dict mapping words to complexity scores (0-1)
    """
    html_parts = []
    for word in sentence.split():
        clean_word = word.strip(".,!?;:")
        score = word_scores.get(clean_word.lower(), 0.0)
        color = get_color(score)
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f'margin: 1px; border-radius: 3px;" title="{score:.2f}">{word}</span>'
        )
    
    html = f'<div style="font-size: 18px; line-height: 2;">{" ".join(html_parts)}</div>'
    display(HTML(html))


def draw_loss_curves(trainer_logs: list[dict], title: str):
    """
    Draws loss curves from trainer.state.log_history

    Params:
        trainer_logs: trainer.state.log_history
    """
    train_steps = []
    train_losses = []

    eval_steps = []
    eval_losses = []

    for entry in trainer_logs:
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])

        if 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])

    plt.figure(figsize=(10, 6))

    # Train loss as line
    plt.plot(train_steps, train_losses, label='Train Loss', alpha=0.8)

    # Eval loss as points only
    plt.scatter(eval_steps, eval_losses, label='Eval Loss', zorder=3, c="orange")

    # Label each eval point with exact loss
    for step, loss in zip(eval_steps, eval_losses):
        plt.annotate(
            f"{loss:.4f}",
            (step, loss),
            textcoords="offset points",
            xytext=(0, 6),
            ha='center',
            fontsize=9
        )

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()