# Illumine Lingao Online Reader

A beautiful, responsive web-based reader for "Illumine Lingao" (临高启明), a Chinese alternate history web novel about 500 modern people who travel back to 1628 Ming Dynasty China.

**Live Reader:** https://lingao.entropydrivenmindset.win

## Features

- 📖 **852 Chapters** - Complete English translation included
- 📱 **Responsive Design** - Works on mobile, tablet, and desktop
- 🎨 **Three Themes** - Light, Sepia, and Dark reading modes
- ⚙️ **Customizable** - Adjustable font size, family, and line height
- 📌 **Progress Tracking** - Saves your reading position automatically
- ⌨️ **Keyboard Shortcuts** - Arrow keys for chapter navigation
- 🔍 **Search** - Search across all chapters

## Deployment

This is a static site that can be deployed to any web host:

### GitHub Pages
1. Push the `docs/` folder to your repository
2. Go to Settings → Pages
3. Select "Deploy from a branch" and choose `/docs`

### Cloudflare Pages / Netlify / Vercel
Upload the contents of the `docs/` folder directly.

## File Structure

```
docs/
├── index.html          # Main reader interface
├── privacy.html        # Privacy policy
├── css/
│   └── style.css       # Stylesheet
├── js/
│   └── reader.js       # Reader application
├── content/            # Chapter data (~33MB)
│   ├── chapters_index.js
│   └── chapters_part_1.js through chapters_part_10.js
└── lingao.jpg          # Cover image
```

## Credits

- **Novel:** Illumine Lingao (临高启明) by Blowing Past the Ear (吹牛者)
- **Translation:** Fan translation project
- **Fonts:** Crimson Text & Inter from Google Fonts

## License

Fan-made reader for the Illumine Lingao project.
