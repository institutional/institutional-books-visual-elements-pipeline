() => {
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.isContentEditable) return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;

    const gallery = document.querySelector("#thumb-strip");
    if (!gallery || gallery.offsetParent === null) return;

    const thumbs = Array.from(gallery.querySelectorAll(".thumbnail-item, .gallery-item, [role='button']"));
    if (!thumbs.length) return;

    const selected = gallery.querySelector(".selected, .gallery-item.selected, [aria-selected='true']");
    let idx = selected ? thumbs.indexOf(selected) : 0;
    if (idx === -1) idx = 0;

    if (e.key === "ArrowLeft") idx = Math.max(0, idx - 1);
    else idx = Math.min(thumbs.length - 1, idx + 1);

    thumbs[idx].click();
    thumbs[idx].scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    e.preventDefault();
  });
}
