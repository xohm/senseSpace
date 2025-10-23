# Student Projects

This directory is dedicated to student projects and experimental code using the SenseSpace framework.

## Purpose

This space provides students with:
- A dedicated area to develop their own applications using SenseSpace
- Freedom to experiment with body tracking data
- Examples and starting points for creative projects
- A place to share code and collaborate

## Getting Started

1. **Create Your Project Folder**
   ```bash
   mkdir students/your_project_name
   cd students/your_project_name
   ```

2. **Import SenseSpace Client**
   ```python
   from senseSpaceLib.senseSpace.client import SenseSpaceClient
   from senseSpaceLib.senseSpace.protocol import Frame
   ```

3. **Start Building**
   - Connect to the SenseSpace server
   - Receive body tracking data
   - Process skeleton information
   - Create your application

## Project Structure

```
students/
├── example_student/       # Reference examples (DO NOT MODIFY)
│   ├── proj1/            # Example: Thick skeleton lines
│   ├── proj2/            # Example: Detection sphere
│   └── proj3/            # Example: Raised arm detection
├── your_project_name/    # Your project folder
│   ├── README.md         # Describe your project
│   ├── main.py           # Your main code
│   ├── requirements.txt  # Additional dependencies
│   └── ...               # Other files
└── README.md             # This file
```

## Example Projects

The `example_student/` folder contains reference implementations:
- **proj1**: Custom skeleton visualization with thick lines and custom colors
- **proj2**: Detection sphere at camera center using QVector3D
- **proj3**: Raised arm detection with interactive sphere size toggle (SPACE key)

**Note**: These examples are for reference only. **Do not modify them**. Instead, create your own project folder and use these as inspiration.

## Guidelines

- **Keep projects self-contained** - Each project should be in its own folder
- **Document your work** - Add a README.md to explain what your project does
- **Share dependencies** - Include a requirements.txt if you use additional packages
- **Credit sources** - If you use code from examples, mention it in your README

## Available Resources

- **Main Examples**: See `/client/examples/` for basic usage patterns
- **Library Documentation**: Check the main README.md for API documentation
- **Server Connection**: Default server runs on `localhost:12345`

## Support

If you need help:
1. Check the main examples in `/client/examples/`
2. Review the library documentation
3. Ask your instructor or peers

## Contribution

Feel free to share interesting projects back with the community. Well-documented and creative projects may be featured in the main examples directory.

---

**Happy coding!** 🚀
